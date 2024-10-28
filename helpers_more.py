
import json, os, time, string, random
import numpy as np

from private_obfuscation.paths import DATADIR, EXPSDIR

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
        'punkt_tab',
        'averaged_perceptron_tagger_eng',
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