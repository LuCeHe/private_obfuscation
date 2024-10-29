import os, zipfile, sys, json

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

import pandas as pd
from private_obfuscation.paths import EXPSDIR, PODATADIR, LOCAL_DATADIR
from private_obfuscation.reformulators import get_reformulation_name
from private_obfuscation.main_ir import all_ds, all_reformulation_types, all_retrievers

add_similarities = True
recreate_the_csv = True

pandas_path = os.path.join(PODATADIR, 'results.csv')
missing_path = os.path.join(LOCAL_DATADIR, 'missing_experiments.json')

if recreate_the_csv:
    dirs = [d for d in os.listdir(EXPSDIR) if 'zip' in d and 'reformulators' in d]

    unzipped = []
    for d in dirs:
        path = os.path.join(EXPSDIR, d)
        if not os.path.exists(path.replace('.zip', '')):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(path.replace('.zip', ''))
        unzipped.append(d.replace('.zip', ''))

    simpath = os.path.join(PODATADIR, 'done_similarities.json')
    with open(simpath, 'r') as f:
        done_similarities = json.load(f)

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
            new_v = v.replace(', 100)', '%100').replace(')', '')
            if 'MonoT5' in new_v:
                new_v = 'BM25%100>>MonoT5'
            exp_dict['name'][k] = new_v
            retriever_indices.append(k)

        rt = exp_dict['reformulation']
        model_name = 'gpt-3.5-turbo' if 'chatgpt3p5' in rt else 'diffpriv'
        model_name = model_name if not rt == 'wordnet' else 'wordnet'

        filename_reformulations = get_reformulation_name(
            model_name, exp_dict['dataset_name'], rt.replace('chatgpt3p5_', '')
        ) if not rt == 'none' else 'original'

        if add_similarities:
            sims = done_similarities[filename_reformulations] if not rt == 'none' else {}
            for k, v in sims.items():
                k_ = f'{k}_similarity'.replace('inter', 'szymkiewicz')
                exp_dict[k_] = v

        metric_names = [k for k in exp_dict.keys() if isinstance(exp_dict[k], dict) and not k == 'name']

        for idx in retriever_indices:
            exp_dict_ = {
                k: v
                for k, v in exp_dict.items() if not k in metric_names and not k == 'name' and not k == 'notes'
            }
            for m in metric_names + ['name']:
                exp_dict_[m] = exp_dict[m][idx]

            exp_dict_['path'] = d
            exp_dicts.append(exp_dict_)

    # cleaning process
    exp_dicts = [
        e for e in exp_dicts if e['dataset_name'] in all_ds
                                and e['reformulation'] in all_reformulation_types
                                and e['retriever'] in all_retrievers
    ]

    print(len(exp_dicts))
    df = pd.DataFrame(exp_dicts)

    # substitute column retriever by name
    df['retriever'] = df['name']

    # remove column name
    df = df.drop(columns=['name'])

    # drop duplicate by keeping the one that has the most recent date in path
    df['date'] = df['path'].apply(lambda x: '-'.join(x.split('--')[:2]))
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d-%H-%M-%S')
    df = df.sort_values('date', ascending=False)
    df = df.drop_duplicates(subset=['dataset_name', 'retriever', 'reformulation'], keep='first')

    print(df.to_string())
    print(df.shape)

    # save pandas
    df.to_csv(pandas_path, index=False)

    # which combinations of data_name, reformulation and retriever are missing
    missing = {'monoT5': [], 'bm25': []}

    # all_retrievers_subset = ['BM25%100>>MonoT5']
    # all_retrievers_subset = ['BM25']
    all_retrievers_subset = ['BM25', 'BM25%100>>MonoT5']

    for retriever in all_retrievers_subset:
        ret = 'monoT5' if 'MonoT5' in retriever else 'bm25'
        for reformulation in all_reformulation_types:
            for dataset_name in all_ds:
                if not any(
                        (df['dataset_name'] == dataset_name)
                        & (df['reformulation'] == reformulation)
                        & (df['retriever'] == retriever)
                ):
                    missing[ret].append({
                        "dataset_name": dataset_name,
                        "reformulation": reformulation,
                        "retriever": ret
                    })

    for k, v in missing.items():
        print(k)
        print(json.dumps(v, indent=2))

    for k, v in missing.items():
        print(f'Number of missing {k} combinations: {len(v)}/{len(all_ds) * len(all_reformulation_types)}')

    # save missing
    with open(missing_path, 'w') as f:
        json.dump(missing, f)
