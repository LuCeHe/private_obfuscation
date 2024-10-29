import os, sys
import warnings
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

import pandas as pd
from private_obfuscation.paths import PODATADIR

metrics_oi_ = ['mean_bert_similarity', 'mean_jaccard_similarity', 'mean_szymkiewicz_similarity', 'map', 'nDCG@10',
               'nDCG@100', 'P@10', 'P@100']
# datasets_oi_ = ['irds:msmarco-document/trec-dl-2020', 'irds:beir/nfcorpus/test']

# load csv
pandas_path = os.path.join(PODATADIR, 'results.csv')
df = pd.read_csv(pandas_path)

# reduce precision of metrics to :.2f
df = df.round(3)

# select only retriever BM25
df = df[df['retriever'] == 'BM25']

all_reformulations = df['reformulation'].unique()
all_datasets = df['dataset_name'].unique()

print("', '".join(all_reformulations))
print(all_datasets)

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


def bold_and_underline(table, metrics_oi, prefix_metric=''):
    pm = prefix_metric

    # split the table between rows to skip and those to treat

    for m in metrics_oi:
        # in bold
        # table[f'{pm}{m}'] = table[f'{pm}{m}'].apply(lambda x: x.replace('\%', '')).astype(float)

        # get best and second best
        table[f'{pm}{m}'] = table[f'{pm}{m}'].astype(float)
        best_metric = table[f'{pm}{m}'].max()
        second_best_metric = table[f'{pm}{m}'].sort_values(ascending=False).iloc[1]

        # in bold
        table[f'{pm}{m}'] = table[f'{pm}{m}'].apply(
            lambda x: f'\\textbf{{{x}}}' if str(x) == str(best_metric) else x
        )
        # underlined
        table[f'{pm}{m}'] = table[f'{pm}{m}'].apply(
            lambda x: f'\\underline{{{x}}}' if str(x) == str(second_best_metric) else x
        )

        # table[f'{pm}{m}'] = table[f'{pm}{m}'].apply(lambda x: x)

    # sort by index
    # table = table.sort_index()

    return table


def create_table(df, refs, datasets_oi, metrics_oi):
    sdf = df.copy()

    rows = []
    row_0 = [''] + [f'd{i}' for i, dataset in enumerate(datasets_oi) for _, _ in enumerate(metrics_oi)]
    rows.append(row_0)
    row_1 = ['reformulator'] + [metric for _ in datasets_oi for metric in metrics_oi]
    rows.append(row_1)

    for ref in refs:
        row = [ref]
        for dataset in datasets_oi:
            dsdf = sdf[(sdf['dataset_name'] == dataset)]
            dsdf = bold_and_underline(dsdf, metrics_oi)
            print(dsdf.head())

            for metric in metrics_oi:
                mask = (dsdf['reformulation'] == ref)
                print(ref, dataset, metric)
                print(dsdf[mask][metric].values[0])
                row.append(f'{dsdf[mask][metric].values[0]}')
        rows.append(row)

    # to pandas and then to latex
    sdf = pd.DataFrame(rows)

    latex = sdf.to_latex(index=False, header=False)

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
        latex = latex.replace(d_string, f'\multicolumn{{{len(metrics_oi)}}}{{c}}{{{d}}} ')

    latex = latex.replace('irds:', '').replace('msmarco-document/', '')
    latex = latex.replace('beir/', '').replace('/test', '')
    latex = latex.replace('trec-dl-2020', 'TREC DL 2020').replace('nfcorpus', 'NF-Corpus')
    latex = latex.replace('scifact', 'SciFact').replace('trec-covid', 'TREC-Covid')
    latex = latex.replace('webis-touche2020/v2', 'Webis-Touche')

    latex = latex.replace('mean_bert_similarity', 'BERT')
    latex = latex.replace('mean_jaccard_similarity', 'Jacc')
    latex = latex.replace('mean_szymkiewicz_similarity', 'Szym')
    latex = latex.replace('map', 'MAP')
    latex = latex.replace('}0', '0}').replace('vik\\textbf{', '\\textbf{vik')

    for i in [1, 5, 50]:
        latex = latex.replace(f'_e{i}', f' $\\epsilon = {i}$')

    latex = latex.replace('$0', '0$').replace('nan', '-')

    return latex


all_latexes = ''
for dataset in tqdm(all_datasets):
    print(f"Generating table for {dataset}")
    latex = create_table(df.copy(), refs, [dataset], metrics_oi_)
    all_latexes += '\n\n' + latex

print(all_latexes)
