import os, random, string, re
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from private_obfuscation.helpers_llms import use_chatgpt
from private_obfuscation.paths import PODATADIR, LOCAL_DATADIR


# Function to obfuscate queries by replacing random characters
def random_obfuscate_query(query):
    chars = list(query)
    num_chars_to_change = len(chars) // 3  # Change 1/3 of the characters
    for _ in range(num_chars_to_change):
        idx = random.randint(0, len(chars) - 1)  # Random index
        chars[idx] = random.choice(string.ascii_letters)  # Replace with random letter
    return ''.join(chars)


def get_obfuscator(obfuscation_type, dataset_name='vaswani'):
    """
    Returns the specified obfuscator.

    :param obfuscation_type: Obfuscation type.
    :return: Obfuscator function.
    """
    if obfuscation_type == "none":
        return lambda query: query

    elif obfuscation_type == "random_char":
        return random_obfuscate_query

    elif obfuscation_type == 'chatgpt_improve':
        return get_saved_llm_obfuscations(dataset_name, 'improve')

    else:
        raise ValueError("Invalid obfuscation type.")


def chatgpt_obfuscator(queries, obfuscation_type):
    if 'improve' in obfuscation_type:
        obfuscations = use_chatgpt(
            personality="You are an expert in Information Retrieval. Reword the query into a very effective version.",
            questions=queries
        )
    else:
        raise ValueError("Invalid obfuscation type.")

    return obfuscations


def create_obfuscations():
    # PyTerrier attempt
    import pyterrier as pt

    # Initialize PyTerrier
    if not pt.started():
        pt.init()

    dataset_name = 'vaswani'
    obfuscation_type = 'improve'

    path = os.path.join(PODATADIR, f"obfuscations_{dataset_name}_{obfuscation_type}.txt")

    if not os.path.exists(path):
        # Load a dataset (use any small available dataset)
        dataset = pt.get_dataset(dataset_name)

        # Get the original topics (queries) from the dataset
        topics = dataset.get_topics()

        # Obfuscate the queries
        topics["original_query"] = topics["query"]

        queries = [row['query'] for index, row in topics.iterrows()]
        obfuscations = chatgpt_obfuscator(queries, obfuscation_type)

        with open(path, 'w') as f:
            f.write(str(obfuscations))

    else:
        with open(path, 'r') as f:
            obfuscations = eval(f.read())

    return obfuscations


def get_saved_llm_obfuscations(dataset_name, obfuscation_type, return_obfuscations=False):
    path = os.path.join(LOCAL_DATADIR, f"obfuscations_{dataset_name}_{obfuscation_type}.txt")
    if not os.path.exists(path):
        path = os.path.join(PODATADIR, f"obfuscations_{dataset_name}_{obfuscation_type}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError("Obfuscations file not found.")
    print('Saved obfuscations path:', path)
    with open(path, 'r') as f:
        obfuscations = eval(f.read())

    def obfuscator(query):
        obfuscation = obfuscations[query].lower()
        obfuscation = re.sub(r'[^\w\s]', '', obfuscation)

        return obfuscation

    if return_obfuscations:
        return obfuscator, obfuscations
    return obfuscator


def obfuscate_queries(topics, obfuscation_type, dataset_name='vaswani'):
    """
    Obfuscates the queries in the topics dataframe.

    :param topics: Topics dataframe.
    :param obfuscation_type: Obfuscation type.
    :return: Obfuscated topics dataframe.
    """
    topics["original_query"] = topics["query"]
    obfuscator = get_obfuscator(obfuscation_type, dataset_name=dataset_name)
    topics["query"] = topics["query"].apply(obfuscator)
    return topics


def obfuscation_distance(sentence1, sentence2, distance_type = 'tfidfcosine'):
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

    else:
        raise ValueError(f"Invalid distance type: {distance_type}")

    return similarity_score


def get_distance_obfuscations(obfuscations):
    # Calculate the distance between the original and obfuscated queries
    distances = []
    print('Calculating obfuscation distances...')
    for query, obfuscated_query in tqdm(obfuscations.items()):
        distance = obfuscation_distance(query, obfuscated_query)
        distances.append(distance)

    mean_distance = sum(distances) / len(distances)
    std_distance = (sum((d - mean_distance) ** 2 for d in distances) / len(distances)) ** 0.5

    print(f"Mean and Std distance between original and obfuscated queries: {mean_distance:.2f} pm {std_distance:.2f}")
    return mean_distance

if __name__ == "__main__":
    # obfuscations = create_obfuscations()
    dataset_name = 'vaswani'
    obfuscation_type = 'improve'
    _, obfuscations = get_saved_llm_obfuscations(dataset_name, obfuscation_type, return_obfuscations=True)
    get_distance_obfuscations(obfuscations)
    # use_chatgpt()
    # obfuscation_distance()
