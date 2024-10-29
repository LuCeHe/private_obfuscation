import json, os, time, string, random
import numpy as np

from private_obfuscation.paths import DATADIR, EXPSDIR


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
    # "improve": "You are an expert in Information Retrieval. Reword the query into a very effective version.",
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

all_reformulation_types = [
    'none',
    # 'random_char',
    # 'chatgpt_improve',
    *[f'chatgpt3p5_{rt}' for rt in list(refs_types.keys())],
    *dp_refs,
    'wordnet',
]
all_retrievers = [
    'bm25',
    # 'colbert',
    'monoT5',
]

all_ds = [
    # 'irds:vaswani',
    # 'irds:beir/arguana',
    'irds:beir/nfcorpus/test',
    'irds:beir/scifact/test',
    'irds:beir/trec-covid',
    'irds:beir/webis-touche2020/v2',
    # 'irds:msmarco-document/trec-dl-2019',
    # 'irds:msmarco-document/trec-dl-2020',
    # 'irds:msmarco-document/trec-dl-2020/judged',
]


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def do_save_dicts(save_dicts, save_dir):
    print('Saving results')
    for key, value in save_dicts.items():
        string_result = json.dumps(value, indent=4, cls=NumpyEncoder)
        path = os.path.join(save_dir, f'{key}.txt')
        with open(path, "w") as f:
            f.write(string_result)


def download_nltks():
    import nltk
    nltk_data_path = os.path.join(DATADIR, 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)

    packages = [
        'stopwords',
        'wordnet',
        'punkt',
        'averaged_perceptron_tagger',
        'universal_tagset',
        # 'punkt_tab',
        # 'averaged_perceptron_tagger_eng',
    ]

    for d in packages:
        try:
            pre_folder = ''
            post_folder = '' if not d == 'wordnet' else '.zip'
            if d == 'stopwords' or d == 'wordnet':
                pre_folder = 'corpora/'
            elif d == 'punkt' or d == 'punkt_tab':
                pre_folder = 'tokenizers/'
            elif (d == 'averaged_perceptron_tagger'
                  or d == 'universal_tagset'
                  or d == 'averaged_perceptron_tagger_eng'):
                pre_folder = 'taggers/'
            nltk.data.find(pre_folder + d + post_folder)
            print(f"{d} tokenizer is already installed.")
        except LookupError:
            print(f"{d} tokenizer not found. Downloading...")
            nltk.download(d, download_dir=nltk_data_path)


def create_exp_dir():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)

    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(5))
    EXPDIR = os.path.join(EXPSDIR, time_string + random_string + '_reformulators')
    os.makedirs(EXPDIR, exist_ok=True)
    return EXPDIR


def create_fake_plots():
    datasets = {'WordNet': 'b', 'ChatGPT': 'orange', 'DiffPriv': 'g'}
    import matplotlib.pyplot as plt

    # 3 subplots horizontally
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    std = .3
    for ax in axs:
        for d, c in datasets.items():
            mean = np.random.rand(2)
            N = 50 if 'GPT' in d else 3 if 'Word' in d else 20
            x = mean[0] + std * np.random.rand(N)
            y = mean[1] + std * np.random.rand(N)
            ax.scatter(x, y, c=c, label=d)

    ax.legend()

    axs[0].set_xlabel('semantic\nsimilarity')
    axs[0].set_ylabel('bm25 MAP')

    axs[1].set_xlabel('privacy\nprotection')
    axs[1].set_ylabel('bm25 MAP')

    axs[2].set_xlabel('semantic\nsimilarity')
    axs[2].set_ylabel('syntacticsimilarity')

    # remove axis
    for ax in axs:
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    plt.show()


if __name__ == "__main__":
    # download_nltks()
    # create_exp_dir()
    create_fake_plots()
    print('Done')
