import os, random, string, re, sys

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

from private_obfuscation.helpers_more import download_nltks

download_nltks()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


stop_words = set(stopwords.words('english'))


def character_similarity(string_1, string_2):
    set_1 = set(string_1.lower())
    set_2 = set(string_2.lower())
    intersection = set_1.intersection(set_2)
    char_sim = len(intersection) / (min(len(set_1), len(set_2)))
    return char_sim


def simplify_sentence(sentence, ps, stop_words):
    sentence = ''.join([c for c in sentence if c.isalpha() or c == ' '])
    sentence = " ".join(sentence.split())
    sentence_set = set([ps.stem(w) for w in word_tokenize(sentence.lower()) if not w in stop_words])
    return sentence_set


def reformulation_similarity(sentence1, sentence2, distance_type='tfidfcosine', kwargs={}):
    # to implement: bm25, bleu, my scan technique, etc.
    # sentence1 = 'measurement of dielectric constant of liquids by the use of microwave techniques'
    # sentence2 = 'How can microwave techniques be utilized to measure the dielectric constant of liquids effectively?'
    # sentence1 with sentence2 gives tfidfcosine of 0.4
    # sentence1 with sentence1 gives tfidfcosine of 1
    # however if:
    # sentence2 = sentence1 + ' which is very nice and cool. I like it so much because it is very nice and cool.'
    # then tfidfcosine of sentence1 with sentence2 is about 0.4 again

    if distance_type == 'tfidfcosine':
        # tf-idf

        # Create a TF-IDF Vectorizer object
        tfidf_vectorizer = TfidfVectorizer()

        # Convert the sentences to their TF-IDF representation
        tfidf_matrix = tfidf_vectorizer.fit_transform([sentence1, sentence2])

        # Calculate the cosine similarity between the two sentences
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    elif distance_type == 'inter':
        ps = kwargs['ps']
        stop_words = kwargs['stop_words']

        # stemmatize and remove stopwords
        set1 = simplify_sentence(sentence1, ps, stop_words)
        set2 = simplify_sentence(sentence2, ps, stop_words)

        z = set1.intersection(set2)
        similarity_score = len(z) / min(len(set1), len(set2))

    else:
        raise ValueError(f"Invalid distance type: {distance_type}")

    return similarity_score


def get_similarity_reformulations(reformulations, distance_type='tfidfcosine'):
    # Calculate the distance between the original and reformulated queries
    distances = []
    print('\nCalculating reformulation distances...')
    kwargs = {}
    if distance_type == 'inter':
        kwargs['ps'] = PorterStemmer()
        kwargs['stop_words'] = stop_words

    for query, reformulated_query in tqdm(reformulations.items()):
        distance = reformulation_similarity(query, reformulated_query, distance_type, kwargs=kwargs)
        distances.append(distance)

    mean_distance = sum(distances) / len(distances)
    std_distance = (sum((d - mean_distance) ** 2 for d in distances) / len(distances)) ** 0.5

    print(
        f"Mean and Std distance between original and reformulated queries: {mean_distance:.2f} pm {std_distance:.2f}")
    return mean_distance


class SimilarityBERT():

    def __init__(self, model_id='bert-base-nli-mean-tokens'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 'sentence-transformers/paraphrase-MiniLM-L6-v2'

        self.cosine_similarity = cosine_similarity
        self.model = SentenceTransformer(model_id, device=device)

    def get_similarity(self, sentences):
        sentence_embeddings = self.model.encode(sentences)
        similarity_scores = self.cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])
        return similarity_scores


def test_small_example():
    sentence1 = 'find a nice restaurant close to Duomo'
    sentence2 = 'look for a good restaurant near Sforzesco'

    kwargs = {}

    kwargs['ps'] = PorterStemmer()
    kwargs['stop_words'] = stop_words
    similarity = reformulation_similarity(sentence1, sentence2, distance_type='inter', kwargs=kwargs)
    print('similarity inter:', similarity)
    similarity = reformulation_similarity(sentence1, sentence2, distance_type='tfidfcosine', kwargs=kwargs)
    print('similarity tfidfcosine:', similarity)


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



similarities_ready = [
    'minilm', 'bert',
    'jaccard', 'tfidf', 'inter',
    'char_jaccard', 'char_tfidf', 'char_inter'
]
class SimilaritiesCalculator():
    def __init__(self, metric_name):
        assert metric_name in similarities_ready, f"Similarity metric {metric_name} not available. Choose from {similarities_ready}."
        self.metric_name = metric_name

        if metric_name == 'minilm':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif metric_name == 'bert':
            self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        else:
            raise ValueError(f"Invalid metric name: {metric_name}")