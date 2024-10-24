import os, random, string, re, sys
from tqdm import tqdm

import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pyterrier as pt
from private_obfuscation.paths import PODATADIR, LOCAL_DATADIR

from private_obfuscation.helpers_llms import use_chatgpt


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

    elif reformulation_type == 'chatgpt_imp rove':
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

        z = set1.intersection(set2)
        similarity_score = len(z) / min(len(set1), len(set2))

    else:
        raise ValueError(f"Invalid distance type: {distance_type}")

    return similarity_score


def get_distance_reformulations(reformulations, distance_type='tfidfcosine'):
    # Calculate the distance between the original and reformulated queries
    distances = []
    print('\nCalculating reformulation distances...')
    kwargs = {}
    if distance_type == 'inter':
        kwargs['ps'] = PorterStemmer()
        kwargs['stop_words'] = stop_words

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

    kwargs['ps'] = PorterStemmer()
    kwargs['stop_words'] = stop_words
    similarity = reformulation_distance(sentence1, sentence2, distance_type='inter', kwargs=kwargs)
    print('similarity inter:', similarity)
    similarity = reformulation_distance(sentence1, sentence2, distance_type='tfidfcosine', kwargs=kwargs)
    print('similarity tfidfcosine:', similarity)


def wordnet_reformulator(query):
    words = query.split()
    reformulated_query = []
    for word in words:
        synsets = wn.synsets(word)
        if synsets:
            reformulated_query.append(synsets[0].lemmas()[0].name())
        else:
            reformulated_query.append(word)
    return ' '.join(reformulated_query)


def wordnet_reformulator_2(query):
    words = query.split()
    reformulated_query = []
    for word in words:
        syns = wn.synonyms(word)

        # flatten
        syns = [syn for synset in syns for syn in synset]
        random.shuffle(syns)
        if syns:
            reformulated_query.append(syns[0])
        else:
            reformulated_query.append(word)
    return ' '.join(reformulated_query).replace('_', ' ')


def wordnet_reformulator_3(query):
    # keep only alphanumeric
    query = ''.join([c for c in query if c.isalpha() or c == ' '])

    words = ' '.join([w for w in query.split() if not w.lower() in stop_words])
    return wordnet_reformulator_2(words)


gensim_available = [
    'fasttext-wiki-news-subwords-300',
    'conceptnet-numberbatch-17-06-300',
    'word2vec-ruscorpora-300',
    'word2vec-google-news-300',
    'glove-wiki-gigaword-50',
    'glove-wiki-gigaword-100',
    'glove-wiki-gigaword-200',
    'glove-wiki-gigaword-300',
    'glove-twitter-25',
    'glove-twitter-50',
    'glove-twitter-100',
    'glove-twitter-200',
    '__testing_word2vec-matrix-synopsis'
]


def character_similarity(string_1, string_2):
    set_1 = set(string_1.lower())
    set_2 = set(string_2.lower())
    intersection = set_1.intersection(set_2)
    char_sim = len(intersection) / (min(len(set_1), len(set_2)))
    return char_sim


class GensimPretrained():
    # https://radimrehurek.com/gensim/models/word2vec.html
    def __init__(self, gensim_name=None):
        if gensim_name in gensim_available:
            import gensim.downloader as api
            self.model = api.load(gensim_name)
        else:
            self.model = None

    def get_similarity(self, sentences, model=None):
        if model is None:
            model = self.model
        return model.n_similarity(sentences[0].split(), sentences[1].split())

    def get_random_similar(self, word, model=None):
        if model is None:
            model = self.model
        try:
            similar_words = model.most_similar(word)
            similar_words = [w for w in similar_words if w[1] > 0.53 and character_similarity(word, w[0]) < .8]
            similar_word = random.choice(similar_words)[0].lower()
        except Exception as e:
            print(e)
            similar_word = ''
        return similar_word

    def get_gensim_reformulator(self, gensim_name=None):
        if isinstance(gensim_name, str):
            import gensim.downloader as api
            model = api.load(gensim_name)
        else:
            model = self.model

        def reformulator(query):
            query = ''.join([c for c in query if c.isalpha() or c == ' '])
            words = [w.lower() for w in query.split() if not w.lower() in stop_words]

            new_query = ' '.join([self.get_random_similar(w, model=model) for w in words]).replace('_', ' ')

            # remove extra spaces
            new_query = ' '.join(new_query.split())

            return new_query

        return reformulator


def wordnet_query_expansion(query):
    expansion = wordnet_reformulator_3(query)
    return f'{query} {expansion}'


def test_reformulators():
    from private_obfuscation.helpers_llms import SimilarityBERT

    kwargs = {}

    kwargs['ps'] = PorterStemmer()
    kwargs['stop_words'] = stop_words

    queries = [
        "I want to know about patient confidentiality",
        "What are the symptoms of diabetes?",
        "How to manage mental health issues"
    ]

    bert_similarity = SimilarityBERT()

    print('Loading reformulators...')
    gp = GensimPretrained()
    reformulators = {
        'wordnet': wordnet_reformulator_3,
        'googlenews': gp.get_gensim_reformulator('word2vec-google-news-300'),
        'glovewiki': gp.get_gensim_reformulator('glove-wiki-gigaword-300'),
        # 'conceptnet': GensimPretrained().get_gensim_reformulator('conceptnet-numberbatch-17-06-300'),
        'glovetwitter': gp.get_gensim_reformulator('glove-twitter-200'),
    }

    print('Testing reformulators...')
    for q in queries:
        print('-' * 50)
        # wn_reformulation = wordnet_reformulator_3(q)
        print('Original:', q)

        for k in reformulators:
            print(f'   {k}')
            try:
                reformulator = reformulators[k]
                reformulation = reformulator(q)
                print(f'     ', reformulation)
                sem_similarity = bert_similarity.get_similarity([q, reformulation])
                assert sem_similarity.shape == (1, 1)
                sem_similarity = sem_similarity[0][0]
                sin_similarity = reformulation_distance(q, reformulation, distance_type='inter', kwargs=kwargs)

                print('   Semantic Similarity: ', sem_similarity, sem_similarity.shape)
                print('   Syntactic Similarity:', sin_similarity)
            except Exception as e:
                print(f'   {k}: Error:', e)

        # print('Wordnet: ', wn_reformulation)
        # sem_similarity = bert_similarity.get_similarity([q, wn_reformulation])
        # assert sem_similarity.shape == (1, 1)
        # sem_similarity = sem_similarity[0][0]
        # sin_similarity = reformulation_distance(q, wn_reformulation, distance_type='inter', kwargs=kwargs)

        # print('Semantic Similarity:', sem_similarity, sem_similarity.shape)
        # print('Syntactic Similarity:', sin_similarity)


class GloveDPreformulator():
    pass


def test_get_glove_vector():
    import numpy as np
    gensim_name = 'glove-twitter-25'
    import gensim.downloader as api
    model = api.load(gensim_name)

    # get vector for car
    gensim_vector = model['car']
    print(gensim_vector)

    # add noise to vector
    noise = 0.1
    noisy_vector = gensim_vector + noise * np.random.normal(size=gensim_vector.shape)

    # get most similar word
    similar_word = model.similar_by_vector(noisy_vector)
    print(similar_word)


def word_net_generalize(sentence='nice cat'):
    words = sentence.split()

    pos = pos_tag(word_tokenize(sentence), tagset='universal')
    print(pos)

    for word in words:
        print('-' * 50)
        syns = wn.synsets(word)
        print(syns)

    for w, p in pos:
        print('-' * 50)
        print(w, p)
        syns = wn.synsets(w, pos=p[0].lower())
        print(syns)
        hypernims = [h.lemmas() for syn in syns for h in syn.hypernyms()]
        holonyms = [h.lemmas() for syn in syns for h in syn.member_holonyms()]
        print('hypernims', hypernims)
        print('holonyms', holonyms)


if __name__ == "__main__":
    # test()
    # test_small_example()
    # test_reformulators()
    # test_get_glove_vector()
    word_net_generalize()
