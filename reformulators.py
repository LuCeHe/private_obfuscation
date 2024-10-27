import os, random, string, re, sys, argparse, json, shutil
from tqdm import tqdm

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

from private_obfuscation.helpers_more import download_nltks
from private_obfuscation.helpers_similarities import reformulation_similarity, get_similarity_reformulations, \
    character_similarity
from private_obfuscation.paths import PODATADIR, LOCAL_DATADIR, DATADIR
from private_obfuscation.helpers_differential_privacy import use_diffpriv_glove, load_glove_embeddings
from private_obfuscation.helpers_llms import use_chatgpt, refs_types, use_huggingface, hf_model_ids, chatgpt_models, \
    dp_refs, llms

download_nltks()

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer

from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
detokenizer = TreebankWordDetokenizer()


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


    else:
        rt = reformulation_type.replace('chatgpt3p5_', '')

        model_name = 'gpt-3.5-turbo' if 'chatgpt3p5' in reformulation_type else None
        model_name = model_name if not rt in dp_refs else 'diffpriv'
        model_name = model_name if not rt == 'wordnet' else 'wordnet'
        return get_saved_reformulations(dataset_name, rt, model_name=model_name)


def get_saved_reformulations(dataset_name, reformulation_type, model_name='wordnet', return_reformulations=False):
    filename = get_reformulation_name(model_name, dataset_name, reformulation_type)
    path = os.path.join(LOCAL_DATADIR, filename)
    print('Saved reformulations path:', path)
    if not os.path.exists(path):
        path = os.path.join(PODATADIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError("reformulations file not found.")

    print('Saved reformulations path:', path)
    with open(path, 'r') as f:
        reformulations = eval(f.read())

    def reformulator(query):
        reformulation = reformulations[query].lower()

        print(f"Original Query:     {query}")
        print(f"reformulated Query: {reformulation}")
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


def test():
    # reformulations = create_reformulations()
    dataset_name = 'vaswani'
    reformulation_type = 'improve'
    _, reformulations = get_saved_reformulations(dataset_name, reformulation_type, return_reformulations=True)

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

    get_similarity_reformulations(reformulations)
    get_similarity_reformulations(reformulations, 'inter')
    # use_chatgpt()
    # reformulation_distance()


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
    from private_obfuscation.helpers_similarities import SimilarityBERT

    kwargs = {}

    kwargs['ps'] = PorterStemmer()
    kwargs['stop_words'] = stop_words

    queries = [
        "I want to know about patient confidentiality",
        "What are the symptoms of diabetes?",
        "How to manage mental health issues",
        "How to prevent heart disease",
        "I have a bishop in my hand",
    ]

    bert_similarity = SimilarityBERT()

    print('Loading reformulators...')
    # gp = GensimPretrained()
    reformulators = {
        'truewordnet': wordnet_generalize,
        'wordnet': wordnet_reformulator_3,
        # 'googlenews': gp.get_gensim_reformulator('word2vec-google-news-300'),
        # 'glovewiki': gp.get_gensim_reformulator('glove-wiki-gigaword-300'),
        # 'conceptnet': GensimPretrained().get_gensim_reformulator('conceptnet-numberbatch-17-06-300'),
        # 'glovetwitter': gp.get_gensim_reformulator('glove-twitter-200'),
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
                sin_similarity = reformulation_similarity(q, reformulation, distance_type='inter', kwargs=kwargs)

                print('   Semantic Similarity: ', sem_similarity, sem_similarity.shape)
                print('   Syntactic Similarity:', sin_similarity)
            except Exception as e:
                print(f'   {k}: Error:', e)


pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they']


def get_hyphol(w, p):
    # print(w, p)
    pos = p[0].lower()
    if p == 'PRON' and w in pronouns:
        return random.choice([p for p in pronouns if p != w])
    elif pos == 'p':
        pos = 'n'
    elif pos == '.':
        return w
    elif pos == 'x' or pos == 'c' or pos == 'd':
        pos = None

    syns = wn.synsets(w, pos=pos)
    hypernims = [h.lemma_names('eng') for syn in syns for h in syn.hypernyms()]
    holonyms = [h.lemma_names('eng') for syn in syns for h in syn.member_holonyms()]
    hyphols = hypernims + holonyms

    new_word = w
    if hyphols:
        flatten = [h for sublist in hyphols for h in sublist if not h == w]
        if flatten:
            new_word = random.choice(flatten)

    return new_word


def wordnet_generalize(sentence='nice cat'):
    pos = pos_tag(word_tokenize(sentence), tagset='universal')
    new_words = [get_hyphol(w, p) for w, p in pos]
    return detokenizer.detokenize(new_words).replace('_', ' ')


def use_wordnet_generalization(
        queries=["What is the capital of France?", "What is the capital of Germany?"],
):
    obfuscations = []
    for query in tqdm(queries):
        print('-' * 50)
        print(query)
        obfuscated_query = wordnet_generalize(query)

        print(obfuscated_query)
        obfuscations.append(obfuscated_query)

    pairs = {query: obfuscated_query for query, obfuscated_query in zip(queries, obfuscations)}
    return pairs


def queries_to_reformulations(queries, reformulation_type, model_name='gpt-3.5-turbo', extra_args=None):
    if reformulation_type in refs_types and model_name in chatgpt_models:
        reformulations = use_chatgpt(
            personality=refs_types[reformulation_type],
            queries=queries,
            api_model=model_name,
        )

    elif reformulation_type in refs_types and model_name in hf_model_ids:
        reformulations = use_huggingface(
            personality=refs_types[reformulation_type],
            queries=queries,
            model_id=model_name,
        )

    elif reformulation_type in dp_refs:
        reformulations = use_diffpriv_glove(
            reformulation_type=reformulation_type,
            queries=queries,
            extra_args=extra_args
        )

    elif reformulation_type == 'wordnet':
        print(f"Generating responses with WordNet Generalization...")
        reformulations = use_wordnet_generalization(queries)
        # reformulations = [wordnet_generalize(q) for q in queries]

    else:
        raise ValueError(f"Model must be one of {llms} for now. Otherwise use Differential Privacy or WordNet.")

    return reformulations


def get_reformulation_name(model_name, dataset_name, reformulation_type):
    dataset_name_ = dataset_name.replace(':', '-').replace('/', '-')
    model_name_ = model_name.replace('/', '-').replace('.', 'p')

    filename = f"reformulations_{model_name_}_{dataset_name_}_{reformulation_type}.txt"

    return filename


def create_reformulations(
        dataset_name='vaswani', reformulation_type='improve', model_name='gpt-3.5-turbo', extra_args=None
):
    # PyTerrier attempt
    import pyterrier as pt

    # Initialize PyTerrier
    if not pt.started():
        pt.init()

    filename = get_reformulation_name(model_name, dataset_name, reformulation_type)
    path = os.path.join(PODATADIR, filename)
    local_path = os.path.join(LOCAL_DATADIR, filename)

    reformulations = None
    if reformulation_type == 'count':
        dataset = pt.get_dataset(dataset_name)

        # Get the original topics (queries) from the dataset
        topics = dataset.get_topics('text')

        print('len queries:', len(topics['query']))
        chars = sum([len(row['query']) for index, row in topics.iterrows()])
        print('      chars:', chars)


    elif not os.path.exists(path) and not os.path.exists(local_path):
        print('Creating reformulations...')
        # Load a dataset (use any small available dataset)
        dataset = pt.get_dataset(dataset_name)

        # Get the original topics (queries) from the dataset
        topics = dataset.get_topics('text')

        print('len queries:', len(topics['query']))

        # Reformulate the queries
        topics["original_query"] = topics["query"]
        queries = [row['query'] for index, row in topics.iterrows()]
        reformulations = queries_to_reformulations(
            queries=queries, reformulation_type=reformulation_type, model_name=model_name,
            extra_args=extra_args
        )

        with open(path, 'w', encoding="utf-8") as f:
            f.write(str(reformulations))

    else:
        if os.path.exists(local_path):
            path = local_path

        print('Loading reformulations...')
        with open(path, 'r') as f:
            reformulations = eval(f.read())

    return reformulations


def main_reformulate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="all")
    parser.add_argument("--reformulation_type", type=str, default="wordnet")
    parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()

    print(json.dumps(args.__dict__, indent=2))

    ds = [
        # 'irds:vaswani',
        'irds:beir/scifact/test',
        'irds:beir/nfcorpus/test',
        'irds:beir/trec-covid',
        'irds:beir/webis-touche2020/v2',
        # 'irds:beir/arguana',
        'irds:msmarco-document/trec-dl-2019',
        'irds:msmarco-document/trec-dl-2020',
    ]
    # reformulation_type = 'prompt1'
    # model_name = 'gpt-3.5-turbo'
    # model_name = 'mamba'

    reforms_llm = ['prompt1', 'prompt2', 'prompt3', 'promptM1k1', 'promptM1k3', 'promptM1k5', 'promptM2k1',
                   'promptM2k3',
                   'promptM2k5', 'promptM3k1', 'promptM3k3', 'promptM3k5']

    reforms = [args.reformulation_type]
    if args.reformulation_type == 'allllm':
        reforms = reforms_llm

    if args.reformulation_type == 'dps':
        reforms = dp_refs

    if not args.dataset_name == 'all':
        ds = [args.dataset_name]

    preload_glove = any([rt in dp_refs for rt in reforms])
    extra_args = {}
    if preload_glove:
        glove_version = '42B'  # '42B' 'glove-twitter-25'
        glove_embeddings = load_glove_embeddings(glove_version)
        extra_args['glove_embeddings'] = glove_embeddings

    print('preload', preload_glove)

    for rt in reforms:
        model_name = args.model_name if not rt in dp_refs else 'diffpriv'
        model_name = model_name if not rt == 'wordnet' else 'wordnet'
        for d in ds:
            print('-' * 50)
            print(f'Dataset: {d}, reformulation: {rt}')
            create_reformulations(dataset_name=d, reformulation_type=rt, model_name=model_name, extra_args=extra_args)


def gpt3_ref_cleaner(reformulation, query, filename):
    clean_reformulation = reformulation
    if 'prompt3' in filename:
        if '\n' in reformulation:
            clean_reformulation = reformulation.split('\n')[-1]

    if 'promptM1k1' in filename or 'promptM2k1' in filename or 'promptM3k1' in filename:
        if ':' in reformulation:
            clean_reformulation = reformulation.split(':')[-1]

    if 'promptM1k3' in filename or 'promptM1k5' in filename or 'promptM2k3' in filename or 'promptM2k5' in filename \
            or 'promptM3k3' in filename or 'promptM3k5' in filename:

        for j in range(6):
            clean_reformulation = clean_reformulation.replace(f'{j}. ', 'x.!')

        clean_reformulation = [r.split('x.!')[-1] for r in clean_reformulation.split('\n')] + [query]
        clean_reformulation = [r[0].upper() + r[1:] for r in clean_reformulation if not r == '']
        clean_reformulation = [r if r[-1] in ['.', '?', '!'] else r + '.' for r in clean_reformulation]

        random.shuffle(clean_reformulation)
        clean_reformulation = ' '.join(clean_reformulation)

    clean_reformulation = clean_reformulation.replace('\"', '').strip()
    return clean_reformulation


def cleaning_gpt3_reformulations():
    # path = os.path.join(PODATADIR, filename)
    dirs = [d for d in os.listdir(PODATADIR) if 'reformulations' in d and 'gpt' in d and 'original' in d]
    print(dirs)

    # d = dirs[0]
    # d = 'reformulations_gpt-3p5-turbo_irds-beir-nfcorpus-test_promptM3k5_original.txt'

    for d in dirs:
        path = os.path.join(PODATADIR, d)
        destination = path.replace('_original.txt', '.txt')

        print(f'Loading reformulations from {d}...')

        with open(path, 'r') as f:
            reformulations = eval(f.read())

        new_reformulations = {}

        for query, reformulation in tqdm(reformulations.items()):
            clean_reformulation = gpt3_ref_cleaner(reformulation, query, d)
            new_reformulations[query] = clean_reformulation

        with open(destination, 'w') as f:
            f.write(str(new_reformulations))


if __name__ == "__main__":
    main_reformulate()
    pass
