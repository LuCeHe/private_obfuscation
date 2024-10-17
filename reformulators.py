import os, random, string, re, sys
from tqdm import tqdm

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

from private_obfuscation.helpers_llms import use_chatgpt
from private_obfuscation.paths import PODATADIR, LOCAL_DATADIR


# Function to reformulate queries by replacing random characters
def random_reformulate_query(query):
    chars = list(query)
    num_chars_to_change = len(chars) // 3  # Change 1/3 of the characters
    for _ in range(num_chars_to_change):
        idx = random.randint(0, len(chars) - 1)  # Random index
        chars[idx] = random.choice(string.ascii_letters)  # Replace with random letter
    return ''.join(chars)


def get_reformulator(reformulation_type, dataset_name='vaswani'):
    """
    Returns the specified reformulator.

    :param reformulation_type: reformulation type.
    :return: reformulator function.
    """
    if reformulation_type == "none":
        return lambda query: query

    elif reformulation_type == "random_char":
        return random_reformulate_query

    elif reformulation_type == 'chatgpt_improve':
        return get_saved_llm_reformulations(dataset_name, 'improve')

    else:
        raise ValueError("Invalid reformulation type.")


def chatgpt_reformulator(queries, reformulation_type):
    if 'improve' in reformulation_type:
        reformulations = use_chatgpt(
            personality="You are an expert in Information Retrieval. Reword the query into a very effective version.",
            questions=queries
        )
    else:
        raise ValueError("Invalid reformulation type.")

    return reformulations


def create_reformulations():
    # PyTerrier attempt
    import pyterrier as pt

    # Initialize PyTerrier
    if not pt.started():
        pt.init()

    dataset_name = 'vaswani'
    reformulation_type = 'improve'

    path = os.path.join(PODATADIR, f"reformulations_{dataset_name}_{reformulation_type}.txt")

    if not os.path.exists(path):
        # Load a dataset (use any small available dataset)
        dataset = pt.get_dataset(dataset_name)

        # Get the original topics (queries) from the dataset
        topics = dataset.get_topics()

        # Reformulate the queries
        topics["original_query"] = topics["query"]

        queries = [row['query'] for index, row in topics.iterrows()]
        reformulations = chatgpt_reformulator(queries, reformulation_type)

        with open(path, 'w') as f:
            f.write(str(reformulations))

    else:
        with open(path, 'r') as f:
            reformulations = eval(f.read())

    return reformulations


def get_saved_llm_reformulations(dataset_name, reformulation_type, return_reformulations=False):
    path = os.path.join(LOCAL_DATADIR, f"obfuscations_{dataset_name}_{reformulation_type}.txt")
    if not os.path.exists(path):
        path = os.path.join(PODATADIR, f"obfuscations_{dataset_name}_{reformulation_type}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError("reformulations file not found.")

    print('Saved reformulations path:', path)
    with open(path, 'r') as f:
        reformulations = eval(f.read())

    def reformulator(query):
        reformulation = reformulations[query].lower()
        reformulation = re.sub(r'[^\w\s]', '', reformulation)

        return reformulation

    if return_reformulations:
        return reformulator, reformulations
    return reformulator


def reformulate_queries(topics, reformulation_type, dataset_name='vaswani'):
    """
    reformulates the queries in the topics dataframe.

    :param topics: Topics dataframe.
    :param reformulation_type: reformulation type.
    :return: reformulated topics dataframe.
    """
    topics["original_query"] = topics["query"]
    reformulator = get_reformulator(reformulation_type, dataset_name=dataset_name)
    topics["query"] = topics["query"].apply(reformulator)
    return topics


def simplify_sentence(sentence, ps, stop_words):
    sentence = ''.join([c for c in sentence if c.isalpha() or c == ' '])
    sentence = " ".join(sentence.split())
    sentence_set = set([ps.stem(w) for w in word_tokenize(sentence.lower()) if not w in stop_words])
    return sentence_set


def reformulation_distance(sentence1, sentence2, distance_type='tfidfcosine', kwargs={}):
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

        # print('set1:', set1)
        # print('set2:', set2)

        z = set1.intersection(set2)

        similarity_score = len(z) / min(len(set1), len(set2))
        # print('similarity_score', similarity_score)

    else:
        raise ValueError(f"Invalid distance type: {distance_type}")

    return similarity_score


def get_distance_reformulations(reformulations, distance_type='tfidfcosine'):
    # Calculate the distance between the original and reformulated queries
    distances = []
    print('\nCalculating reformulation distances...')
    kwargs = {}
    if distance_type == 'inter':
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt')

        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        from nltk.stem import PorterStemmer

        kwargs['ps'] = PorterStemmer()
        kwargs['stop_words'] = set(stopwords.words('english'))

    for query, reformulated_query in tqdm(reformulations.items()):
        distance = reformulation_distance(query, reformulated_query, distance_type, kwargs=kwargs)
        distances.append(distance)

    mean_distance = sum(distances) / len(distances)
    std_distance = (sum((d - mean_distance) ** 2 for d in distances) / len(distances)) ** 0.5

    print(
        f"Mean and Std distance between original and reformulated queries: {mean_distance:.2f} pm {std_distance:.2f}")
    return mean_distance



def test():
    # reformulations = create_reformulations()
    dataset_name = 'vaswani'
    reformulation_type = 'improve'
    _, reformulations = get_saved_llm_reformulations(dataset_name, reformulation_type, return_reformulations=True)

    i = 0
    for query, reformulated_query in reformulations.items():
        print('-' * 50)
        print(f"Original Query:   {query}")
        reformulation = re.sub(r'[^\w\s]', '', reformulated_query).lower()
        print(f"reformulated Query: {reformulation}")
        i += 1
        if i > 5:
            break
    print('-' * 50)

    get_distance_reformulations(reformulations)
    get_distance_reformulations(reformulations, 'inter')
    # use_chatgpt()
    # reformulation_distance()

def test_small_example():
    sentence1 = 'find a nice restaurant close to Duomo'
    sentence2 = 'look for a good restaurant near Sforzesco'

    kwargs = {}
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

    from nltk.corpus import stopwords

    from nltk.stem import PorterStemmer

    kwargs['ps'] = PorterStemmer()
    kwargs['stop_words'] = set(stopwords.words('english'))
    similarity = reformulation_distance(sentence1, sentence2, distance_type='inter', kwargs=kwargs)
    print('similarity inter:', similarity)
    similarity = reformulation_distance(sentence1, sentence2, distance_type='tfidfcosine', kwargs=kwargs)
    print('similarity tfidfcosine:', similarity)

if __name__ == "__main__":
    test()
    # test_small_example()

