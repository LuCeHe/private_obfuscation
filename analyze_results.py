import argparse, os, zipfile, sys, json

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

import pandas as pd
from private_obfuscation.paths import EXPSDIR

dirs = [d for d in os.listdir(EXPSDIR) if 'zip' in d and 'reformulators' in d]
print(dirs)

unzipped = []
for d in dirs:
    path = os.path.join(EXPSDIR, d)
    if not os.path.exists(path.replace('.zip', '')):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(EXPSDIR)
    else:
        unzipped.append(d.replace('.zip', ''))

print('experiments done:', len(unzipped))

# d = unzipped[0]

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
