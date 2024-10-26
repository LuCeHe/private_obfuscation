# -*- coding: utf-8 -*-
"""ECIR_Updated_DP_Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SIqM6CfIY3Cuja5NCZ50yFKwpzLqcuYN
"""

from tqdm import tqdm

import numpy as np
import pandas as pd
import numpy.random as npr
from scipy.linalg import sqrtm
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

from private_obfuscation.helpers_more import download_nltks
from private_obfuscation.paths import DATADIR, PODATADIR

download_nltks()

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize

detokenizer = TreebankWordDetokenizer()


def euclidean_distance_matrix(x, y):
    x_expanded = x[:, np.newaxis, :]
    y_expanded = y[np.newaxis, :, :]
    return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))


class AbstractMechanism:
    def __init__(self, m, epsilon=50, **kwargs):
        self.epsilon = epsilon
        self.m = m

    def noise_sampling(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_protected_vectors(self, embeddings):
        os.environ["OMP_NUM_THREADS"] = "1"
        noisy_embeddings = []
        for e in embeddings:
            noise = self.noise_sampling()
            noisy_embeddings.append(e + noise)
        return np.array(noisy_embeddings)


# CMP Mechanism class
class CMPMechanism(AbstractMechanism):
    def __init__(self, m, epsilon=50, **kwargs):
        super().__init__(m, epsilon, **kwargs)

    def noise_sampling(self):
        N = npr.multivariate_normal(np.zeros(self.m), np.eye(self.m))
        X = N / np.sqrt(np.sum(N ** 2))
        Y = npr.gamma(self.m, 1 / self.epsilon)
        Z = Y * X
        return Z


# MM class
class MahalanobisMechanism(AbstractMechanism):
    def __init__(self, m, epsilon=50, embeddings=None, lam=1, **kwargs):
        super().__init__(m, epsilon, **kwargs)
        self.lam = lam
        self.emb_matrix = embeddings
        _, self.m = self.emb_matrix.shape
        cov_mat = np.cov(self.emb_matrix.T, ddof=0)
        self.sigma = cov_mat / np.mean(np.var(self.emb_matrix.T, axis=1))
        self.sigma_loc = sqrtm(self.lam * self.sigma + (1 - self.lam) * np.eye(self.m))

    def noise_sampling(self):
        N = npr.multivariate_normal(np.zeros(self.m), np.eye(self.m))
        X = N / np.sqrt(np.sum(N ** 2))
        X = np.matmul(self.sigma_loc, X)
        X = X / np.sqrt(np.sum(X ** 2))
        Y = npr.gamma(self.m, 1 / self.epsilon)
        Z = X * Y
        return Z


# VKM class
class VickreyMechanism(MahalanobisMechanism):
    def __init__(self, m, epsilon=50, **kwargs):
        super().__init__(m, epsilon, **kwargs)
        self.lam = kwargs.get('lambda', 0.75)

    def get_protected_vectors(self, embeddings):
        n_words = len(embeddings)
        noisy_embeddings = []
        for e in embeddings:
            noisy_embeddings.append(e + self.noise_sampling())

        noisy_embeddings = np.array(noisy_embeddings)
        distance = euclidean_distance_matrix(noisy_embeddings, self.emb_matrix)

        closest = np.argpartition(distance, 2, axis=1)[:, :2]
        dist_to_closest = distance[np.tile(np.arange(n_words).reshape(-1, 1), 2), closest]

        p = ((1 - self.lam) * dist_to_closest[:, 1]) / (
                self.lam * dist_to_closest[:, 0] + (1 - self.lam) * dist_to_closest[:, 1])

        vickrey_choice = np.array([npr.choice(2, p=[p[w], 1 - p[w]]) for w in range(n_words)])
        noisy_embeddings = self.emb_matrix[closest[np.arange(n_words), vickrey_choice]]

        return noisy_embeddings


# VKMM class
class VickreyMMechanism(MahalanobisMechanism):
    def __init__(self, m, epsilon=50, **kwargs):
        super().__init__(m, epsilon, **kwargs)
        self.lam = kwargs.get('lambda', 0.75)

    def get_protected_vectors(self, embeddings):
        n_words = len(embeddings)
        noisy_embeddings = []
        for e in embeddings:
            noisy_embeddings.append(e + self.noise_sampling())

        noisy_embeddings = np.array(noisy_embeddings)
        distance = euclidean_distance_matrix(noisy_embeddings, self.emb_matrix)

        closest = np.argpartition(distance, 2, axis=1)[:, :2]
        dist_to_closest = distance[np.tile(np.arange(n_words).reshape(-1, 1), 2), closest]

        p = ((1 - self.lam) * dist_to_closest[:, 1]) / (
                self.lam * dist_to_closest[:, 0] + (1 - self.lam) * dist_to_closest[:, 1])

        vickrey_choice = np.array([npr.choice(2, p=[p[w], 1 - p[w]]) for w in range(n_words)])
        noisy_embeddings = self.emb_matrix[closest[np.arange(n_words), vickrey_choice]]

        return noisy_embeddings


# VKCM class
class VickreyCMPMechanism(MahalanobisMechanism):
    def __init__(self, m, epsilon=50, embeddings=None, lam=0.75, **kwargs):
        super().__init__(m, epsilon, embeddings=embeddings, lam=lam, **kwargs)
        self.cmp_mechanism = CMPMechanism(m, epsilon)

    def get_protected_vectors(self, embeddings):
        n_words = len(embeddings)

        noisy_embeddings = []
        for e in embeddings:
            noise = self.cmp_mechanism.noise_sampling()
            noisy_embeddings.append(e + noise)
        noisy_embeddings = np.array(noisy_embeddings)

        distance = euclidean_distance_matrix(noisy_embeddings, self.emb_matrix)
        closest = np.argpartition(distance, 2, axis=1)[:, :2]
        dist_to_closest = distance[np.tile(np.arange(n_words).reshape(-1, 1), 2), closest]

        p = ((1 - self.lam) * dist_to_closest[:, 1]) / (
                self.lam * dist_to_closest[:, 0] + (1 - self.lam) * dist_to_closest[:, 1])

        vickrey_choice = np.array([npr.choice(2, p=[p[w], 1 - p[w]]) for w in range(n_words)])
        noisy_embeddings = self.emb_matrix[closest[np.arange(n_words), vickrey_choice]]

        return noisy_embeddings


# synthetic data generation based on embedding matrix
# def generate_synthetic_data(n_samples, embedding_dim):
#     embeddings = npr.randn(n_samples, embedding_dim)
#     df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embedding_dim)])
#     return df


# Query simulation function
def simulate_queries(n_queries, embedding_dim):
    queries = npr.randn(n_queries, embedding_dim)
    query_df = pd.DataFrame(queries, columns=[f"dim_{i}" for i in range(embedding_dim)])
    return query_df


def visualize_embeddings(original, protected_cmp, protected_mah, protected_vickrey):
    plt.figure(figsize=(15, 5))

    # Plot original embeddings
    plt.subplot(2, 2, 1)
    plt.scatter(original[:, 0], original[:, 1], alpha=0.5, c='blue', label='Original Embeddings')
    plt.title('Original Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.legend()

    # CMP protection
    plt.subplot(2, 2, 2)
    plt.scatter(protected_cmp[:, 0], protected_cmp[:, 1], alpha=0.5, c='red', label='Protected CMP')
    plt.title('Protected Embeddings (CMP)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.legend()

    # Mahalanobis protection embeddings
    plt.subplot(2, 2, 3)
    plt.scatter(protected_mah[:, 0], protected_mah[:, 1], alpha=0.5, c='green', label='Protected Mahalanobis')
    plt.title('Protected Embeddings (Mahalanobis)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.legend()

    # plt.tight_layout()
    # plt.show()

    # Vickrey protection embeddings
    plt.subplot(2, 2, 4)
    plt.scatter(protected_vickrey[:, 0], protected_vickrey[:, 1], alpha=0.5, c='yellow', label='Protected Vickrey')
    plt.title('Protected Embeddings (Vickrey)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def load_data():
    from gensim.models import KeyedVectors

    base_dir = DATADIR
    path = os.path.join(base_dir, 'glove-twitter-25', 'glove-twitter-25.gz')
    model = KeyedVectors.load_word2vec_format(path)
    return model


def load_glove_embeddings(gensim_name='glove-twitter-25'):
    import gensim.downloader as api
    embeddings_dict = api.load(gensim_name)

    return embeddings_dict


def find_closest_words(embedding, glove_embeddings, top_n=10):
    """Finds the closest words to a given embedding in the GloVe vocabulary."""
    distances = {
        word: np.linalg.norm(embedding - glove_embeddings[word])
        for word in glove_embeddings
    }
    closest_words = sorted(distances, key=distances.get)[:top_n]
    return closest_words


def get_glove_vector(glove_embeddings, word):
    if not word in glove_embeddings:
        # get the most similar word that is in the dictionary
        similarities = {
            w: get_jacard_similarity(set(list(word)), set(list(w)))
            for w in glove_embeddings.index_to_key
        }
        most_similar_word = max(similarities, key=similarities.get)
        word = most_similar_word
    vector = glove_embeddings[word]
    return vector


def obfuscate_text(text, mechanism, glove_embeddings):
    """Obfuscates text on a per-word basis using the given DP mechanism."""

    # tokenize
    words = word_tokenize(text.lower())
    embeddings = [get_glove_vector(glove_embeddings, word) for word in words]

    if not embeddings:
        return ""

    embeddings = np.array(embeddings)

    protected_embeddings = mechanism.get_protected_vectors(embeddings)
    obfuscated_words = [glove_embeddings.similar_by_vector(emb)[0][0] for emb in protected_embeddings]
    obfuscated_sentence = detokenizer.detokenize(obfuscated_words)
    return obfuscated_sentence


def get_jacard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0 if set1 and set2 else np.nan


def calculate_similarities(original_query, obfuscated_query, semantic_model):
    """Calculates Jaccard and sentence similarities."""

    original_words = set(word_tokenize(original_query.lower()))
    obfuscated_words = set(word_tokenize(obfuscated_query.lower()))
    jaccard_similarity = get_jacard_similarity(original_words, obfuscated_words)

    embedding1 = semantic_model.encode(original_query, convert_to_tensor=True)
    embedding2 = semantic_model.encode(obfuscated_query, convert_to_tensor=True)
    semantic_similarity = util.cos_sim(embedding1, embedding2).item()

    return jaccard_similarity, semantic_similarity


def use_diffpriv_glove(
        reformulation_type='vikcmp_e1',
        queries=["What is the capital of France?", "What is the capital of Germany?"],
        extra_args=None
):
    if not extra_args is None and 'glove_embeddings' in extra_args:
        glove_embeddings = extra_args['glove_embeddings']
    else:
        glove_embeddings = load_glove_embeddings()

    embedding_dim = glove_embeddings.vector_size

    glove_matrix = np.array([glove_embeddings[word] for word in glove_embeddings.index_to_key])

    epsilon = int(reformulation_type.split('_e')[-1])
    if reformulation_type.startswith('cmp'):
        mech = CMPMechanism(m=embedding_dim, epsilon=epsilon)
    elif reformulation_type.startswith('mah'):
        mech = MahalanobisMechanism(m=embedding_dim, epsilon=epsilon, embeddings=glove_matrix)
    elif reformulation_type.startswith('vik_'):
        mech = VickreyMechanism(m=embedding_dim, epsilon=epsilon, embeddings=glove_matrix)
    elif reformulation_type.startswith('vikm_'):
        mech = VickreyMMechanism(m=embedding_dim, epsilon=epsilon, embeddings=glove_matrix)
    elif reformulation_type.startswith('vikcmp_'):
        mech = VickreyCMPMechanism(m=embedding_dim, epsilon=epsilon, embeddings=glove_matrix)
    else:
        raise ValueError(f"Unknown mechanism type: {reformulation_type}")

    obfuscations = []
    print(f"Generating responses with Differential Privacy ({reformulation_type})...")
    for query in tqdm(queries):
        obfuscated_query = obfuscate_text(query, mech, glove_embeddings)

        print(obfuscated_query)
        obfuscations.append(obfuscated_query)

    pairs = {query: obfuscated_query for query, obfuscated_query in zip(queries, obfuscations)}
    return pairs


def test_diffpriv():
    glove_embeddings = load_glove_embeddings()
    embedding_dim = glove_embeddings.vector_size

    queries = [
        "What is the prognosis for endocarditis?",
        "What are the symptoms of diabetes?",
        "How to treat hypertension?",
        "What are the risk factors for heart disease?",
        "Is there a cure for Alzheimer's disease?"
    ]

    glove_matrix = np.array([glove_embeddings[word] for word in glove_embeddings.index_to_key])

    epsilon = 50
    mechs = {
        "CMP": CMPMechanism(m=embedding_dim, epsilon=epsilon),
        "Mahalanobis": MahalanobisMechanism(m=embedding_dim, epsilon=epsilon, embeddings=glove_matrix),
        "Vickrey": VickreyMechanism(m=embedding_dim, epsilon=epsilon, embeddings=glove_matrix),
        "VickreyM": VickreyMMechanism(m=embedding_dim, epsilon=epsilon, embeddings=glove_matrix),
        "VickreyCMP": VickreyCMPMechanism(m=embedding_dim, epsilon=epsilon, embeddings=glove_matrix)
    }
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    for i, original_query in enumerate(queries):
        print(f"\nOriginal Query {i + 1}: {original_query}")

        for mech_name, mech in mechs.items():
            print('   Mechanism:', mech_name)
            obfuscated_query = obfuscate_text(original_query, mech, glove_embeddings)
            print(f"   Obfuscated Query: {obfuscated_query}")
            jaccard_sim, semantic_sim = calculate_similarities(original_query, obfuscated_query, semantic_model)
            print(f"{mech_name} - Jaccard: {jaccard_sim:.4f}, Semantic: {semantic_sim:.4f}")


def load_glove_model_42B():
    path = os.path.join(PODATADIR, 'glove_commoncrawl_42B', 'glove.42B.300d.txt')
    print("Loading Glove Model")
    glove_model = {}
    with open(path, 'r', encoding="utf8") as f:
        for line in tqdm(f, total=1917494):
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
            assert embedding.shape == (300,), f"Shape is {embedding.shape} for word {word}"
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def load_glove_model_42B_gensim():
    import requests
    import zipfile

    glove_commoncrawl_42B_folder = os.path.join(PODATADIR, 'glove_commoncrawl_42B')
    os.makedirs(glove_commoncrawl_42B_folder, exist_ok=True)

    # download this zip https://nlp.stanford.edu/data/glove.42B.300d.zip
    if not os.path.exists(os.path.join(glove_commoncrawl_42B_folder, 'glove.42B.300d.txt')):
        # Define the URL and the local path to save the file
        url = "https://nlp.stanford.edu/data/glove.42B.300d.zip"
        # zip_path = "glove.42B.300d.zip"
        zip_path = os.path.join(glove_commoncrawl_42B_folder, 'glove.42B.300d.zip')

        # Download the file
        print("Starting download...")
        response = requests.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the file content to disk in chunks
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print("Download complete.")
        else:
            print(f"Failed to download. Status code: {response.status_code}")

        # Extract the downloaded zip file
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("glove_42B")
        print("Extraction complete.")




    # print("Loading Glove Model with Gensim")
    #
    # import gensim
    # import time
    # from gensim.models import KeyedVectors
    #
    # start_time = time.time()
    # # Specify the path to your GloVe file
    # glove_file = 'path/to/glove.42B.300d.txt'
    # path = os.path.join(glove_commoncrawl_42B_folder, 'glove.42B.300d.txt')
    #
    # # Convert GloVe file format to a Gensim KeyedVectors format
    # glove_model = KeyedVectors.load_word2vec_format(path, binary=False, no_header=True)
    #
    # cat_vector = glove_model['cat']
    # print(cat_vector)
    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # test_diffpriv()
    # load_glove_model_42B()
    load_glove_model_42B_gensim()
