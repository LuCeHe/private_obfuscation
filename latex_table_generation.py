import os, sys

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

import pandas as pd
from private_obfuscation.paths import PODATADIR

# load csv
pandas_path = os.path.join(PODATADIR, 'results.csv')
df = pd.read_csv(pandas_path)
print(df.head())

# select only retriever BM25
df = df[df['retriever'] == 'BM25']
print(df.head())
print(list(df.columns))

all_reformulations = df['reformulation'].unique()
all_datasets = df['dataset_name'].unique()

print("', '".join(all_reformulations))
print(all_datasets)

metrics_oi = ['mean_bert_similarity', 'mean_jaccard_similarity', 'mean_szymkiewicz_similarity', 'map', 'nDCG@10']
datasets_oi = ['irds:msmarco-document/trec-dl-2020', 'irds:beir/nfcorpus/test']

# new columns: reformulation,

# row one
# ['',              data_one*len(metrics_oi),  data_two*len(metrics_oi), ...]
# ['',              d1,         d1,            d2,         d2            ...]
# ['reformulation', 'metric_1', 'metric_2',    'metric_1', 'metric_2',   ...]


refs = [
    'wordnet',
    'cmp_e1', 'cmp_e5', 'cmp_e50',
    'mah_e1', 'mah_e5', 'mah_e50',
    'vik_e1', 'vik_e5', 'vik_e50',
    'vikcmp_e1', 'vikcmp_e5', 'vikcmp_e50',
    'vikm_e1', 'vikm_e5', 'vikm_e50',
    'chatgpt3p5_prompt1', 'chatgpt3p5_prompt2', 'chatgpt3p5_prompt3',
    'chatgpt3p5_promptM1k1', 'chatgpt3p5_promptM1k3', 'chatgpt3p5_promptM1k5',
    'chatgpt3p5_promptM2k1', 'chatgpt3p5_promptM2k3', 'chatgpt3p5_promptM2k5',
    'chatgpt3p5_promptM3k1', 'chatgpt3p5_promptM3k3', 'chatgpt3p5_promptM3k5',
    'none',
]

rows = []
row_0 = [''] + [f'd{i}' for i, dataset in enumerate(datasets_oi) for j, metric in enumerate(metrics_oi)]
rows.append(row_0)
row_1 = ['reformulator'] + [metric for dataset in datasets_oi for metric in metrics_oi]
rows.append(row_1)

for ref in refs:
    row = [ref]
    for dataset in datasets_oi:
        for metric in metrics_oi:
            mask = (df['dataset_name'] == dataset) & (df['reformulation'] == ref)
            row.append(df[mask][metric].mean())
    rows.append(row)

print(rows)

# to pandas and then to latex
df = pd.DataFrame(rows)

latex = df.to_latex(index=False, header=False)

latex = f"""
\\begin{{table}}[H]
\\centering
\\resizebox{{\columnwidth}}{{!}}{{%
{latex}
}}
\\caption{{Effect that different obfuscations have.}}
\\label{{tab:table1}}
\\end{{table}}
"""

for ref in refs:
    latex = latex.replace(ref, f'\\textbf{{{ref}}}')

# cleaning names
latex = latex.replace('chatgpt3p5_prompt', 'GPT p')
latex = latex.replace('wordnet', 'WordNet')
latex = latex.replace('reformulator', '\\textbf{reformulator}')

for m in metrics_oi:
    latex = latex.replace(m, f'\\textbf{{{m}}}')

for i, d in enumerate(datasets_oi):
    d_string = ' & '.join([f'd{i}'] * len(metrics_oi))
    print(d_string)
    latex = latex.replace(d_string, f'\multicolumn{{{len(metrics_oi)}}}{{c}}{{{d}}} ')

latex = latex.replace('irds:', '').replace('msmarco-document/', '')
latex = latex.replace('beir/', '').replace('/test', '')
latex = latex.replace('trec-dl-2020', 'TREC DL 2020').replace('nfcorpus', 'NF-Corpus')

latex = latex.replace('mean_bert_similarity', 'BERT')
latex = latex.replace('mean_jaccard_similarity', 'Jaccard')
latex = latex.replace('mean_szymkiewicz_similarity', 'Szymkiewicz')
latex = latex.replace('_e', ' $\\epsilon =$')

print(latex)
