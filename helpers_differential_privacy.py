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
from sentence_transformers import SentenceTransformer

from private_obfuscation.helpers_more import download_nltks
from private_obfuscation.helpers_similarities import get_jacard_similarity, calculate_similarities
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


def vickrey_get_protected_vectors(self, embeddings):
    n_words = len(embeddings)
    noisy_embeddings = []
    for e in embeddings:
        noisy_embeddings.append(e + self.noise_sampling())
        # noisy_embeddings.append(e)

    noisy_embeddings = np.array(noisy_embeddings)
    distance = euclidean_distance_matrix(noisy_embeddings, self.emb_matrix)

    closest = np.argpartition(distance, 2, axis=1)[:, :2]
    dist_to_closest = distance[np.tile(np.arange(n_words).reshape(-1, 1), 2), closest]

    p = ((1 - self.lam) * dist_to_closest[:, 1]
         / (self.lam * dist_to_closest[:, 0] + (1 - self.lam) * dist_to_closest[:, 1]))

    vickrey_choice = np.array([npr.choice(2, p=[p[w], 1 - p[w]]) for w in range(n_words)])
    noisy_embeddings = self.emb_matrix[closest[np.arange(n_words), vickrey_choice]]

    return noisy_embeddings


# VKM class
class VickreyMechanism(MahalanobisMechanism):
    def __init__(self, m, epsilon=50, **kwargs):
        super().__init__(m, epsilon, **kwargs)
        self.lam = kwargs.get('lambda', 0.75)

    def get_protected_vectors(self, embeddings):
        return vickrey_get_protected_vectors(self, embeddings)


# VKMM class
class VickreyMMechanism(MahalanobisMechanism):
    def __init__(self, m, epsilon=50, **kwargs):
        super().__init__(m, epsilon, **kwargs)
        self.lam = kwargs.get('lambda', 0.75)

    def get_protected_vectors(self, embeddings):
        return vickrey_get_protected_vectors(self, embeddings)


# VKCM class
class VickreyCMPMechanism(MahalanobisMechanism):
    def __init__(self, m, epsilon=50, embeddings=None, lam=0.75, **kwargs):
        super().__init__(m, epsilon, embeddings=embeddings, lam=lam, **kwargs)
        self.cmp_mechanism = CMPMechanism(m, epsilon)

    def get_protected_vectors(self, embeddings):
        return vickrey_get_protected_vectors(self, embeddings)


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


def load_glove_embeddings(gensim_name='42B'):
    if gensim_name == '42B':
        return load_glove_model_42B_gensim()

    # 'glove-twitter-25'
    import gensim.downloader as api
    embeddings_dict = api.load(gensim_name)

    return embeddings_dict


def load_glove_model_42B_gensim():
    import requests
    import zipfile

    import gensim
    import time
    from gensim.models import KeyedVectors

    glove_commoncrawl_42B_folder = os.path.join(PODATADIR, 'glove_commoncrawl_42B')
    txt_path = os.path.join(glove_commoncrawl_42B_folder, 'glove.42B.300d.txt')
    os.makedirs(glove_commoncrawl_42B_folder, exist_ok=True)

    # download this zip https://nlp.stanford.edu/data/glove.42B.300d.zip
    if not os.path.exists(txt_path):
        # Define the URL and the local path to save the file
        url = "https://nlp.stanford.edu/data/glove.42B.300d.zip"
        # zip_path = "glove.42B.300d.zip"
        zip_path = os.path.join(glove_commoncrawl_42B_folder, 'glove.42B.300d.zip')

        if not os.path.exists(zip_path):
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
            zip_ref.extractall(glove_commoncrawl_42B_folder)
        print("Extraction complete.")

    print("Loading Glove Model with Gensim")

    start_time = time.time()
    # Specify the path to your GloVe file
    glove_file = 'path/to/glove.42B.300d.txt'
    path = os.path.join(glove_commoncrawl_42B_folder, 'glove.42B.300d.txt')

    # Convert GloVe file format to a Gensim KeyedVectors format
    glove_model = KeyedVectors.load_word2vec_format(path, binary=False, no_header=True)

    print("--- %s seconds ---" % (time.time() - start_time))

    return glove_model


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
    return obfuscated_sentence.replace('_', ' ')



def obfuscate_text_memory_safer(text, mechanism, glove_embeddings):
    """Obfuscates text on a per-word basis using the given DP mechanism."""

    # tokenize
    words = word_tokenize(text.lower())
    protected_embeddings = np.array([
        mechanism.get_protected_vectors(
            get_glove_vector(glove_embeddings, word)[None]
        )[0] for word in words
    ])

    obfuscated_words = [glove_embeddings.similar_by_vector(emb)[0][0] for emb in protected_embeddings]
    obfuscated_sentence = detokenizer.detokenize(obfuscated_words)
    return obfuscated_sentence.replace('_', ' ')

def get_dp_mech(reformulation_type, embedding_dim, epsilon, glove_matrix):
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
    return mech


def use_diffpriv_glove(
        reformulation_type='vikcmp_e1',
        queries=["What is the capital of France?", "What is the capital of Germany?"],
        extra_args=None
):
    import string, random, time
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)

    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(5))

    tmppath = os.path.join(PODATADIR, time_string + random_string + f'_{reformulation_type}_tmp_reformulations.txt')

    if not extra_args is None and 'glove_embeddings' in extra_args:
        glove_embeddings = extra_args['glove_embeddings']
    else:
        glove_embeddings = load_glove_embeddings()

    embedding_dim = glove_embeddings.vector_size

    glove_matrix = np.array([glove_embeddings[word] for word in glove_embeddings.index_to_key])

    epsilon = int(reformulation_type.split('_e')[-1])
    mech = get_dp_mech(reformulation_type, embedding_dim, epsilon, glove_matrix)

    print(f"Generating responses with Differential Privacy ({reformulation_type})...")

    for i, query in tqdm(enumerate(queries)):
        if i % 100 and not i == 0:
            del mech
            mech = get_dp_mech(reformulation_type, embedding_dim, epsilon, glove_matrix)

        # obfuscated_query = obfuscate_text(query, mech, glove_embeddings)
        obfuscated_query = obfuscate_text_memory_safer(query, mech, glove_embeddings)

        with open(tmppath, 'a', encoding='utf-8') as f:
            f.write(f"{query}###DIVIDEUNLIKELY####{obfuscated_query}\n")

    with open(tmppath, 'r', encoding='utf-8') as f:
        obfuscations = f.readlines()

    obfuscations = [line.strip().split('###DIVIDEUNLIKELY####')[1] for line in obfuscations]

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


def test_vickrey():
    epsilon = 500
    vocab = 20
    n_words = 5
    e_dim = 4

    mock_glove = np.random.rand(vocab, e_dim)
    mechanism = VickreyMechanism(m=e_dim, epsilon=epsilon, embeddings=mock_glove)

    embeddings = [npr.randn(e_dim) for _ in range(n_words)]
    print('Embeddings:', embeddings)

    protected_embeddings = np.array([mechanism.get_protected_vectors(e[None])[0] for e in embeddings])
    print('Protected embeddings:', protected_embeddings)

    embeddings = np.array(embeddings)
    print('embeddings.shape: ', embeddings.shape)

    protected_embeddings = mechanism.get_protected_vectors(embeddings)
    print('Protected embeddings:', protected_embeddings)


def test_use_diffpriv():
    glove_version = 'glove-twitter-25'  # '42B' 'glove-twitter-25'
    glove_embeddings = load_glove_embeddings(glove_version)
    refs = use_diffpriv_glove(
        'vikcmp_e3',
        [
            "What is the capital of France?",
            "What is the capital of Germany?",
            "Another sentence"
        ],
        extra_args={'glove_embeddings': glove_embeddings}
    )
    print(refs)

if __name__ == "__main__":
    test_use_diffpriv()
    # test_vickrey()

    pass
