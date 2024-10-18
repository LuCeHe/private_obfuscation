# -*- coding: utf-8 -*-
"""dp_evaluation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nRmcCEvtbxKub-zKX4EcbemiHtMfGU5j
"""

import numpy as np
import pandas as pd
import numpy.random as npr
from scipy.linalg import sqrtm
from sklearn.metrics import jaccard_score
import os

# Abstract base mechanism class
class AbstractMechanism:
    def __init__(self, m, epsilon=1, **kwargs):
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
    def __init__(self, m, epsilon=1, **kwargs):
        super().__init__(m, epsilon, **kwargs)

    def noise_sampling(self):
        N = npr.multivariate_normal(np.zeros(self.m), np.eye(self.m))
        X = N / np.sqrt(np.sum(N ** 2))
        Y = npr.gamma(self.m, 1 / self.epsilon)
        Z = Y * X
        return Z

# Mahalanobis Mechanism class
class MahalanobisMechanism(AbstractMechanism):
    """
    Xu, Zekun, Abhinav Aggarwal, Oluwaseyi Feyisetan, and Nathanael Teissier. "A Differentially Private Text Perturbation Method Using Regularized
    Mahalanobis Metric." In Proceedings of the Second Workshop on Privacy in NLP, pp. 7-17. 2020.
    """
    def __init__(self, m, epsilon=1, embeddings=None, lam=1, **kwargs):
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

# Vickrey MMechanism class
class VickreyMMechanism(MahalanobisMechanism):
    """
    Zekun Xu, Abhinav Aggarwal, Oluwaseyi Feyisetan, Nathanael Teissier: On a Utilitarian Approach to Privacy Preserving Text Generation.
    CoRR abs/2104.11838 (2021)
    """

    def __init__(self, m, epsilon=1, **kwargs):
        super().__init__(m, epsilon, **kwargs)
        self.lam = kwargs.get('lambda', 0.75)

    def get_protected_vectors(self, embeddings):
        n_words = len(embeddings)
        noisy_embeddings = []
        for e in embeddings:
            noisy_embeddings.append(e + self.noise_sampling())

        def euclidean_distance_matrix(x, y):
            x_expanded = x[:, np.newaxis, :]
            y_expanded = y[np.newaxis, :, :]

            return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))

        noisy_embeddings = np.array(noisy_embeddings)
        distance = euclidean_distance_matrix(noisy_embeddings, self.emb_matrix)

        closest = np.argpartition(distance, 2, axis=1)[:, :2]
        dist_to_closest = distance[np.tile(np.arange(n_words).reshape(-1, 1), 2), closest]

        p = ((1 - self.lam) * dist_to_closest[:, 1]) / (self.lam * dist_to_closest[:, 0] + (1 - self.lam) * dist_to_closest[:, 1])

        vickrey_choice = np.array([npr.choice(2, p=[p[w], 1 - p[w]]) for w in range(n_words)])
        noisy_embeddings = self.emb_matrix[closest[np.arange(n_words), vickrey_choice]]

        return noisy_embeddings

class VickreyCMPMechanism(MahalanobisMechanism):

    """
    Zekun Xu, Abhinav Aggarwal, Oluwaseyi Feyisetan, Nathanael Teissier: On a Utilitarian Approach to Privacy Preserving Text Generation.
    CoRR abs/2104.11838 (2021)

    This class combines the Vickrey and CMP mechanisms.
    It first applies the CMP mechanism to add noise to the embeddings.
    Then, it uses the Vickrey mechanism to select between the original
    embedding and a nearby embedding from the synthetic data.
    """

    def __init__(self, m, epsilon=1, embeddings=None, lam=0.75, **kwargs):
        super().__init__(m, epsilon, embeddings=embeddings, lam=lam, **kwargs)
        # Initialization of the CMP mechanism
        self.cmp_mechanism = CMPMechanism(m, epsilon)

    def get_protected_vectors(self, embeddings):
        n_words = len(embeddings)

        # Stage 1: Apply the CMP mechanism to add noise
        noisy_embeddings = []
        for e in embeddings:
            # The CMP is used to add noise
            noise = self.cmp_mechanism.noise_sampling()
            noisy_embeddings.append(e + noise)
        noisy_embeddings = np.array(noisy_embeddings)

        # Stage 2: Apply Vickrey mechanism for selection
        distance = self.euclidean_distance_matrix(noisy_embeddings, self.emb_matrix)
        closest = np.argpartition(distance, 2, axis=1)[:, :2]
        dist_to_closest = distance[np.tile(np.arange(n_words).reshape(-1, 1), 2), closest]

        p = ((1 - self.lam) * dist_to_closest[:, 1]) / (self.lam * dist_to_closest[:, 0] + (1 - self.lam) * dist_to_closest[:, 1])

        vickrey_choice = np.array([npr.choice(2, p=[p[w], 1 - p[w]]) for w in range(n_words)])
        noisy_embeddings = self.emb_matrix[closest[np.arange(n_words), vickrey_choice]]

        return noisy_embeddings

    #Helper function to calculate Euclidean distance matrix
    def euclidean_distance_matrix(self, x, y):
        x_expanded = x[:, np.newaxis, :]
        y_expanded = y[np.newaxis, :, :]
        return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))

#Vickrey Mechanism class
class VickreyMechanism(MahalanobisMechanism):
    """
    Zekun Xu, Abhinav Aggarwal, Oluwaseyi Feyisetan, Nathanael Teissier: On a Utilitarian Approach to Privacy Preserving Text Generation.
    CoRR abs/2104.11838 (2021)
    """

    def __init__(self, m, epsilon=1, **kwargs):
        super().__init__(m, epsilon, **kwargs)
        self.lam = kwargs.get('lambda', 0.75)

    def get_protected_vectors(self, embeddings):
        n_words = len(embeddings)
        noisy_embeddings = []
        for e in embeddings:
            noisy_embeddings.append(e + self.noise_sampling())

        def euclidean_distance_matrix(x, y):
            x_expanded = x[:, np.newaxis, :]
            y_expanded = y[np.newaxis, :, :]

            return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))

        noisy_embeddings = np.array(noisy_embeddings)
        distance = euclidean_distance_matrix(noisy_embeddings, self.emb_matrix)

        closest = np.argpartition(distance, 2, axis=1)[:, :2]
        dist_to_closest = distance[np.tile(np.arange(n_words).reshape(-1, 1), 2), closest]

        p = ((1 - self.lam) * dist_to_closest[:, 1]) / (self.lam * dist_to_closest[:, 0] + (1 - self.lam) * dist_to_closest[:, 1])

        vickrey_choice = np.array([npr.choice(2, p=[p[w], 1 - p[w]]) for w in range(n_words)])
        noisy_embeddings = self.emb_matrix[closest[np.arange(n_words), vickrey_choice]]

        return noisy_embeddings

#Generation of the synthetic data based on embedding matrix
def generate_synthetic_data(n_samples, embedding_dim):
    embeddings = npr.randn(n_samples, embedding_dim)
    df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embedding_dim)])
    return df

# Functin for the query simulation
def simulate_queries(n_queries, embedding_dim):
    queries = npr.randn(n_queries, embedding_dim)
    query_df = pd.DataFrame(queries, columns=[f"dim_{i}" for i in range(embedding_dim)])
    return query_df

#Function to compute all the DP mechanisms on the queries
def run_dp_mechanism(queries, mechanism_class, embedding_dim, epsilon, embeddings=None):
    dp_mechanism = mechanism_class(m=embedding_dim, epsilon=epsilon, embeddings=embeddings)
    noisy_embeddings = dp_mechanism.get_protected_vectors(queries.values)

    noisy_queries_df = pd.DataFrame(noisy_embeddings, columns=[f"dim_{i}" for i in range(embedding_dim)])

    return noisy_queries_df

def precision_at_k(relevant_items, retrieved_items, k):
    relevant_set = set(tuple(row) for row in relevant_items)
    retrieved_set = set(tuple(row) for row in retrieved_items[:k])
    common_items = relevant_set.intersection(retrieved_set)
    return len(common_items) / k

def recall(relevant_items, retrieved_items, k):
    relevant_set = set(tuple(row) for row in relevant_items)
    retrieved_set = set(tuple(row) for row in retrieved_items[:k])
    common_items = relevant_set.intersection(retrieved_set)
    return len(common_items) / len(relevant_set) if relevant_set else 0

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

#Evaluation of the query results
def evaluate_query(query_embeddings, noisy_embeddings, synthetic_data, k=10):
    if query_embeddings.shape[1] != synthetic_data.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: queries have {query_embeddings.shape[1]} while synthetic_data has {synthetic_data.shape[1]} dimensions")

    query_embeddings_array = query_embeddings.values
    noisy_embeddings_array = noisy_embeddings.values
    synthetic_data_array = synthetic_data.values


    #Calculates the Euclidean distance between each query embedding and all the embeddings in the synthetic data.
    distance_matrix = np.linalg.norm(query_embeddings_array[:, np.newaxis, :] - synthetic_data_array, axis=2)


    """
    Retrieves the indices of the top-k items based on the calculated distances
    for example, retrieving the top-k items based on distance
    """
    top_k_retrieved_indices = np.argsort(distance_matrix, axis=1)[:, :k]

    # We have to define the closest items relevant to the original queries
    relevant_items = np.argsort(distance_matrix, axis=1)[:, :k]

    precision = precision_at_k(relevant_items, top_k_retrieved_indices, k)
    rec = recall(relevant_items, top_k_retrieved_indices, k)
    f1 = f1_score(precision, rec)

    return precision, rec, f1, top_k_retrieved_indices

# Function to calculate Jaccard similarity
def calculate_jaccard_similarity(original_queries, obfuscated_queries):
    similarities = []

    for i in range(len(original_queries)):
        original_queries_binary = (original_queries[i] > 0).astype(int)
        obfuscated_queries_binary = (obfuscated_queries[i] > 0).astype(int)
        similarity = jaccard_score(original_queries_binary, obfuscated_queries_binary)
        similarities.append(similarity)

    return np.mean(similarities)

# Function to evaluate all the DP mechanism
def evaluate_all_models(corpus, queries, synthetic_data, epsilon):
    results = {}


    mechanisms = {
        "CMP": CMPMechanism,
        "Mahalanobis": MahalanobisMechanism,
        "VickreyCMP": VickreyCMPMechanism,
        "Vickrey": VickreyMechanism,
        "VickreyM": VickreyMMechanism
    }

    for name, mechanism in mechanisms.items():
        if name in ["Mahalanobis", "VickreyCMP", "Vickrey", "VickreyM"]:
            dp_protected_queries = run_dp_mechanism(queries, mechanism, embedding_dim, epsilon, embeddings=synthetic_data.values)
        else:
            dp_protected_queries = run_dp_mechanism(queries, mechanism, embedding_dim, epsilon)

        precision, rec, f1, _ = evaluate_query(queries, dp_protected_queries, synthetic_data)

        # We also evaluated the Jaccard similarity
        jaccard_sim = calculate_jaccard_similarity(queries.values, dp_protected_queries.values)

        results[name] = {
            'Precision': precision,
            'Recall': rec,
            'F1 Score': f1,
            'Jaccard Similarity': jaccard_sim
        }

    return results

#A synthetic dataset and queries for testing were generated

# We decided to use 2000 samples in the synthetic dataset
n_samples = 2000

# It is important to set the embedding dimension to ensure that it matches the synthetic data dimension for consistency
embedding_dim = 1024
synthetic_data = generate_synthetic_data(n_samples, embedding_dim)

# Since we are working on multiple query scenarios, let 20 be the mumber of queries to simulate
n_queries = 20

# The queries must match the synthetic data dimension
queries = simulate_queries(n_queries, embedding_dim)


epsilon = 0.5
results = evaluate_all_models(synthetic_data, queries, synthetic_data, epsilon)

for model, metrics in results.items():
    print(f"Results for {model}: Precision: {metrics['Precision']}, Recall: {metrics['Recall']}, F1 Score: {metrics['F1 Score']}, Jaccard Similarity: {metrics['Jaccard Similarity']:.4f}")