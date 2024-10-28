import argparse, os, zipfile, sys, json

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

import numpy as np
import pandas as pd
from private_obfuscation.paths import EXPSDIR, PODATADIR
from private_obfuscation.reformulators import get_reformulation_name

dirs = [d for d in os.listdir(EXPSDIR) if 'zip' in d and 'reformulators' in d]

unzipped = []
for d in dirs:
    path = os.path.join(EXPSDIR, d)
    if not os.path.exists(path.replace('.zip', '')):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(EXPSDIR)
    else:
        unzipped.append(d.replace('.zip', ''))


simpath = os.path.join(PODATADIR, 'done_similarities.json')
with open(simpath, 'r') as f:
    done_similarities = json.load(f)
print('sim keys', done_similarities.keys())

exp_dicts = []
for d in unzipped:
    files = os.listdir(os.path.join(EXPSDIR, d))

    exp_dict = {}
    for file in files:
        with open(os.path.join(EXPSDIR, d, file), 'r') as f:
            exp_dict.update(json.load(f))

    retriever_indices = []
    for k, v in exp_dict['name'].items():
        v = v.replace('RankCutoff(', '').replace('TerrierRetr(', '')
        exp_dict['name'][k] = v.replace(', 100)', '%100').replace(')', '')
        retriever_indices.append(k)

    rt = exp_dict['reformulation']
    model_name = 'gpt-3.5-turbo' if 'chatgpt3p5' in rt else 'diffpriv'
    model_name = model_name if not rt == 'wordnet' else 'wordnet'

    filename_reformulations = get_reformulation_name(
        model_name, exp_dict['dataset_name'], rt.replace('chatgpt3p5_', '')
    ) if not rt == 'none' else 'original'

    sims = done_similarities[filename_reformulations] if not rt == 'none' else {}
    for k, v in sims.items():
        k_ = f'{k}_similarity'.replace('inter', 'szymkiewicz')
        exp_dict[f'mean_{k_}'] = np.mean(v)
        exp_dict[f'std_{k_}'] = np.std(v)

    metric_names = [k for k in exp_dict.keys() if isinstance(exp_dict[k], dict) and not k == 'name']

    for idx in retriever_indices:
        exp_dict_ = {
            k: v
            for k, v in exp_dict.items() if not k in metric_names and not k == 'name'
        }
        for m in metric_names + ['name']:
            exp_dict_[m] = exp_dict[m][idx]

        exp_dict_['path'] = d
        exp_dicts.append(exp_dict_)

print(len(exp_dicts))
df = pd.DataFrame(exp_dicts)
print(df.to_string())
print(df.shape)


# save pandas
pandas_path = os.path.join(PODATADIR, 'results.csv')
df.to_csv(pandas_path, index=False)
