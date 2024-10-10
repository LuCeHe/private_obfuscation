import os, random, string
from tqdm import tqdm

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))
DATADIR = os.path.abspath(os.path.join(WORKDIR, "data", "private_obfuscation"))
os.makedirs(DATADIR, exist_ok=True)


# Function to obfuscate queries by replacing random characters
def random_obfuscate_query(query):
    chars = list(query)
    num_chars_to_change = len(chars) // 3  # Change 1/3 of the characters
    for _ in range(num_chars_to_change):
        idx = random.randint(0, len(chars) - 1)  # Random index
        chars[idx] = random.choice(string.ascii_letters)  # Replace with random letter
    return ''.join(chars)


def get_obfuscator(obfuscation_type):
    """
    Returns the specified obfuscator.

    :param obfuscation_type: Obfuscation type.
    :return: Obfuscator function.
    """
    if obfuscation_type == "none":
        return lambda query: query

    elif obfuscation_type == "random_char":
        return random_obfuscate_query

    else:
        raise ValueError("Invalid obfuscation type.")


def chatgpt_obfuscator(queries, obfuscation_type):
    if 'improve' in obfuscation_type:
        obfuscations = use_chatgpt(
            personality="You are an expert in Information Retrieval. Reword the query into a very effective version.",
            questions=queries
        )
    else:
        obfuscations = {query: obfuscate_query(query) for query in queries}

    return obfuscations


def save_obfuscations():
    # PyTerrier attempt
    import pyterrier as pt

    # Initialize PyTerrier
    if not pt.started():
        pt.init()

    dataset_name = 'vaswani'
    obfuscation_type = 'improve'

    path = os.path.join(DATADIR, f"obfuscations_{dataset_name}_{obfuscation_type}.txt")

    if not os.path.exists(path):
        # Load a dataset (use any small available dataset)
        dataset = pt.get_dataset(dataset_name)

        # Get the original topics (queries) from the dataset
        topics = dataset.get_topics()

        # Obfuscate the queries
        topics["original_query"] = topics["query"]
        print(topics)

        i = 0
        obfuscations = {}
        queries = [row['query'] for index, row in topics.iterrows()]
        obfuscations = chatgpt_obfuscator(queries, obfuscation_type)

        with open(path, 'w') as f:
            f.write(str(obfuscations))

    else:
        with open(path, 'r') as f:
            obfuscations = eval(f.read())
    print(obfuscations)


def get_saved_obfuscations(dataset_name, obfuscation_type):
    path = os.path.join(DATADIR, f"obfuscations_{dataset_name}_{obfuscation_type}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError("Obfuscations file not found.")
    with open(path, 'r') as f:
        obfuscations = eval(f.read())

    def obfuscator(query):
        return obfuscations[query]
    return obfuscator

def use_chatgpt(
        personality="You are a helpful assistant that translates English to French. Translate the user sentence.",
        questions=["What is the capital of France?", "What is the capital of Germany?"],
):
    os.path.join(WORKDIR, 'all_stuff.py', )

    from all_stuff import api_key_openai

    # os.environ["OPENAI_API_KEY"] = api_key_openai

    from langchain_openai import ChatOpenAI

    api_model = 'gpt-3.5-turbo'
    llm = ChatOpenAI(
        model=api_model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key_openai,
    )

    answers = []
    print("Generating responses with ChatGPT...")
    for question in tqdm(questions):
        messages = [
            ("system", personality),
            ("human", question),
        ]
        ai_msg = llm.invoke(messages).content
        print(ai_msg)
        answers.append(ai_msg)

    pairs = {question: answer for question, answer in zip(questions, answers)}
    return pairs


def obfuscate_queries(topics, obfuscation_type):
    """
    Obfuscates the queries in the topics dataframe.

    :param topics: Topics dataframe.
    :param obfuscation_type: Obfuscation type.
    :return: Obfuscated topics dataframe.
    """
    topics["original_query"] = topics["query"]
    obfuscator = get_obfuscator(obfuscation_type)
    topics["query"] = topics["query"].apply(obfuscator)
    return topics

if __name__ == "__main__":
    save_obfuscations()
    # use_chatgpt()
