import os, sys
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

from private_obfuscation.helpers_differential_privacy import use_diffpriv_glove, load_glove_embeddings
from private_obfuscation.paths import WORKDIR, DATADIR, PODATADIR, LOCAL_DATADIR

hf_model_ids = {
    'mamba': 'state-spaces/mamba-130m-hf',
    'mambaxl': 'state-spaces/mamba-2.8b-hf',
    'falconmamba': 'tiiuae/falcon-mamba-7b-instruct',
    'gpt2': 'openai-community/gpt2',
    'gpt2xl': 'openai-community/gpt2-xl',
    'llama3': 'meta-llama/Llama-3.2-1B',
    'falcon': 'tiiuae/falcon-7b',
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'internlm': 'internlm/internlm2_5-7b-chat',
}

chatgpt_models = [
    'gpt-3.5-turbo',
]

llms = chatgpt_models + list(hf_model_ids.keys())

refs_types = {
    "improve": "You are an expert in Information Retrieval. Reword the query into a very effective version.",
    "prompt1": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Rewrite this query.",
    "prompt2": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Rewrite this query by applying the concept of generalization.",
    "prompt3": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Rewrite this query by applying the concept of differential privacy.",
    "promptM1k1": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Generate 1 dummy, random queries, given this query.",
    "promptM1k3": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Generate 3 dummy, random queries, given this query.",
    "promptM1k5": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Generate 5 dummy, random queries, given this query.",
    "promptM2k1": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Generate 1 dummy queries, which are semantically related to this query.",
    "promptM2k3": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Generate 3 dummy queries, which are semantically related to this query.",
    "promptM2k5": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Generate 5 dummy queries, which are semantically related to this query.",
    "promptM3k1": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Generate 1 dummy queries, which generalize this query.",
    "promptM3k3": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Generate 3 dummy queries, which generalize this query.",
    "promptM3k5": "You are a privacy expert tasked with protecting the user's privacy when performing the following task. Generate 5 dummy queries, which generalize this query.",
}

dp_refs = [
    "cmp_e1", "cmp_e5", "cmp_e50",
    "mah_e1", "mah_e5", "mah_e50",
    "vik_e1", "vik_e5", "vik_e50",
    "vikm_e1", "vikm_e5", "vikm_e50",
    "vikcmp_e1", "vikcmp_e5", "vikcmp_e50",
]


def use_chatgpt(
        personality="You are a helpful assistant that translates English to French. Translate the user sentence.",
        queries=["What is the capital of France?", "What is the capital of Germany?"],
        api_model='gpt-3.5-turbo',
):
    os.path.join(WORKDIR, 'all_stuff.py', )

    from all_stuff import api_key_openai

    # os.environ["OPENAI_API_KEY"] = api_key_openai

    from langchain_openai import ChatOpenAI

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
    for question in tqdm(queries):
        messages = [
            ("system", personality),
            ("human", question),
        ]
        ai_msg = llm.invoke(messages).content
        print('question', question)
        print('reply', ai_msg)
        answers.append(ai_msg)

    pairs = {question: answer for question, answer in zip(queries, answers)}
    return pairs


def get_model(model_id, device, more_kwargs={}, tkn_kwargs={}):
    'export HF_HOME=/scratch/gpfs/$USER/.cache/huggingface'
    'export HG_DATASETS_CACHE=/scratch/gpfs/$USER/.cache/huggingface/datasets'
    os.environ["HF_HOME"] = DATADIR
    os.environ["HG_DATASETS_CACHE"] = DATADIR
    os.system(f'export HF_HOME={DATADIR}')
    os.system(f'export HG_DATASETS_CACHE={DATADIR}')

    datadir = DATADIR
    # datadir = '/home/lucacehe/.cache/huggingface/hub'
    # datadir = os.path.join(os.getenv("HOME"), '.cache', 'huggingface', 'hub')
    path_model = os.path.join(datadir, 'models--' + model_id.replace('/', '--'))
    path_tokenizer = os.path.join(datadir, 'tokenizers--' + model_id.replace('/', '--'))
    print('path:', path_model)
    print('   does exist?', os.path.exists(path_model))
    if not os.path.exists(path_model):
        print('Downloading model')
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, truncation_side='left', **tkn_kwargs
        )

        tokenizer.save_pretrained(path_tokenizer)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, **more_kwargs
        )

        model.save_pretrained(path_model)
    else:
        print('Loading model')
        tokenizer = AutoTokenizer.from_pretrained(path_tokenizer, truncation_side='left', **tkn_kwargs)
        model = AutoModelForCausalLM.from_pretrained(path_model, **more_kwargs)

    return model.to(device), tokenizer


def use_huggingface(
        personality="You are a helpful assistant that translates English to French. Translate the USER SENTENCE. USER SENTENCE:",
        queries=["What is the capital of France?", "What is the capital of Germany?"],
        model_id='mambaxl'
):
    model_id = hf_model_ids[model_id]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = get_model(model_id, device)

    # inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")
    # outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    # reply = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # print(reply)

    answers = []
    print(f"Generating responses with {model_id}...")
    for question in tqdm(queries):
        print('-' * 50)
        full_question = f"{personality} {question}. SOLUTION:".replace('..', '.').replace('?.', '?')

        inputs = tokenizer(full_question, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'mamba' in model_id:
            del inputs['attention_mask']
        outputs = model.generate(**inputs, do_sample=True, top_p=0.2, max_new_tokens=40)
        reply = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print('full_question:', full_question)
        print('reply:        ', reply)
        answers.append(reply)

    pairs = {question: answer for question, answer in zip(queries, answers)}
    return pairs


class SimilarityBERT():

    def __init__(self, model_id='bert-base-nli-mean-tokens'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 'sentence-transformers/paraphrase-MiniLM-L6-v2'
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        self.cosine_similarity = cosine_similarity
        self.model = SentenceTransformer(model_id, device=device)

    def get_similarity(self, sentences):
        sentence_embeddings = self.model.encode(sentences)
        similarity_scores = self.cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])
        return similarity_scores


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

    else:
        if not model_name in llms:
            raise ValueError(f"Model must be one of {llms} for now")
        else:
            raise ValueError(f"Reformulation type must be one of {refs_types}")

    return reformulations


def create_reformulations(
        dataset_name='vaswani', reformulation_type='improve', model_name='gpt-3.5-turbo', extra_args=None
):
    # PyTerrier attempt
    import pyterrier as pt

    # Initialize PyTerrier
    if not pt.started():
        pt.init()

    dataset_name_ = dataset_name.replace(':', '-').replace('/', '-')
    model_name_ = model_name.replace('/', '-').replace('.', 'p')

    filename = f"reformulations_{model_name_}_{dataset_name_}_{reformulation_type}.txt"
    path = os.path.join(PODATADIR, filename)
    local_path = os.path.join(LOCAL_DATADIR, filename)

    reformulator = queries_to_reformulations if model_name in llms or reformulation_type in dp_refs else None
    assert reformulator is not None, f"Model must be one of {llms} for now or use Differential Privacy"

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
        reformulations = reformulator(
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


if __name__ == '__main__':
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="irds:msmarco-document/trec-dl-2020")
    parser.add_argument("--reformulation_type", type=str, default="vikm_e1")
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

    reforms = ['prompt1', 'prompt2', 'prompt3', 'promptM1k1', 'promptM1k3', 'promptM1k5', 'promptM2k1', 'promptM2k3',
               'promptM2k5', 'promptM3k1', 'promptM3k3', 'promptM3k5']

    if not args.reformulation_type == 'all':
        reforms = [args.reformulation_type]

    if args.reformulation_type == 'dps':
        reforms = dp_refs

    if not args.dataset_name == 'all':
        ds = [args.dataset_name]

    preload_glove = any([rt in dp_refs for rt in reforms])
    extra_args = {}
    if preload_glove:
        glove_version = 'glove-twitter-25' # '42B'
        glove_embeddings = load_glove_embeddings(glove_version)
        extra_args['glove_embeddings'] = glove_embeddings

    print('preload', preload_glove)

    for rt in reforms:
        model_name = args.model_name if not rt in dp_refs else 'diffpriv'
        for d in ds:
            print('-' * 50)
            print(f'Dataset: {d}, reformulation: {rt}')
            create_reformulations(dataset_name=d, reformulation_type=rt, model_name=model_name, extra_args=extra_args)
