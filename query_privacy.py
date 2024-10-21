from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def sensitive_terms_masking(query, sensitive_dict):
    for term, mask in sensitive_dict.items():
        query = query.replace(term, mask)
    return query


def sparse_vec(reformulated_queries):
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(reformulated_queries)
    return tfidf_vectors.toarray()


def dense_vec(tokenized_queries):
    word2vec_model = Word2Vec(tokenized_queries, vector_size=50, min_count=1, workers=2)

    dense_vectors = []
    for tokens in tokenized_queries:
        vector = np.mean([word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv], axis=0)
        dense_vectors.append(vector)

    return np.array(dense_vectors)


def cal_mse_mae(original_vectors, protected_vectors):
    mse = mean_squared_error(original_vectors, protected_vectors)
    mae = mean_absolute_error(original_vectors, protected_vectors)
    return mse, mae


def privacy_loss_ev(original_query, protected_query, sensitive_dict):
    exposed_terms = 0
    total_sensitive = len(sensitive_dict)

    for term in sensitive_dict.keys():
        if term in original_query and term in protected_query:
            exposed_terms += 1

    return exposed_terms / total_sensitive * 100


def privacy_score_ev(original_query, protected_query, sensitive_dict):
    protected_terms = 0
    total_sensitive = len(sensitive_dict)

    for term in sensitive_dict.keys():
        if term in original_query and term not in protected_query:
            protected_terms += 1

    return protected_terms / total_sensitive * 100


def privacy_metrics_evaluation(original_queries, protected_queries, sensitive_dict):
    privacy_loss_scores = []
    privacy_scores = []

    for original, protected in zip(original_queries, protected_queries):
        privacy_loss = privacy_loss_ev(original, protected, sensitive_dict)
        privacy_score = privacy_score_ev(original, protected, sensitive_dict)
        privacy_loss_scores.append(privacy_loss)
        privacy_scores.append(privacy_score)

    return np.mean(privacy_loss_scores), np.mean(privacy_scores)


def reformulate_evaluate(queries, sensitive_dict):
    reformulated_queries = [sensitive_terms_masking(query, sensitive_dict) for query in queries]

    print('Original Queries:    ', queries)
    print('Reformulated Queries:', reformulated_queries)

    sparse_vectors = sparse_vec(reformulated_queries)
    tokenized_queries = [query.split() for query in reformulated_queries]
    dense_vectors = dense_vec(tokenized_queries)
    original_sparse_vectors = np.random.rand(len(queries), sparse_vectors.shape[1])
    original_dense_vectors = np.random.rand(len(queries), dense_vectors.shape[1])

    mse_sparse, mae_sparse = cal_mse_mae(original_sparse_vectors, sparse_vectors)
    mse_dense, mae_dense = cal_mse_mae(original_dense_vectors, dense_vectors)

    privacy_loss, privacy_score = privacy_metrics_evaluation(queries, reformulated_queries, sensitive_dict)

    print(f"Sparse Model Evaluation - MSE: {mse_sparse}, MAE: {mae_sparse}")
    print(f"Dense Model Evaluation - MSE: {mse_dense}, MAE: {mae_dense}")
    print(f"Privacy Loss (%): {privacy_loss}")
    print(f"Privacy Score (%): {privacy_score}")


if __name__ == "__main__":
    sensitive_terms = {
        "my house address": "USER_ADDRESS",
        "Sesto,  20099": "Via Peace",
        "credit card number": "CREDIT_CARD",
        "1234 5678 9012 3456": "CARD_001"
    }

    queries = [
        "My house address is Sesto, 20099.",
        "Please send the package to Via Peace 123.",
        "My credit card number is 1234 5678 9012 3456."
    ]

    reformulate_evaluate(queries, sensitive_terms)
