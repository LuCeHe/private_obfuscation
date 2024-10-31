import os, sys
import warnings
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

import pandas as pd
from private_obfuscation.paths import PODATADIR

metrics_oi_ = [
    'map', 'nDCG@10', 'nDCG@100',
    'mean_bert_similarity', 'mean_jaccard_similarity',
    # 'mean_szymkiewicz_similarity',
]
# datasets_oi_ = ['irds:msmarco-document/trec-dl-2020', 'irds:beir/nfcorpus/test']

# load csv
pandas_path = os.path.join(PODATADIR, 'results.csv')
df = pd.read_csv(pandas_path)

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
    'none',
    'wordnet',
    'cmp_e1', 'cmp_e5', 'cmp_e10', 'cmp_e50',
    'mah_e1', 'mah_e5', 'mah_e10', 'mah_e50',
    'vik_e1', 'vik_e5', 'vik_e10', 'vik_e50',
    # 'vikcmp_e1', 'vikcmp_e5', 'vikcmp_e10', 'vikcmp_e50',
    # 'vikm_e1', 'vikm_e5', 'vikm_e10', 'vikm_e50',
    'chatgpt3p5_prompt1', 'chatgpt3p5_prompt2', 'chatgpt3p5_prompt3',
    'chatgpt3p5_promptM1k1', 'chatgpt3p5_promptM1k3', 'chatgpt3p5_promptM1k5',
    'chatgpt3p5_promptM2k1', 'chatgpt3p5_promptM2k3', 'chatgpt3p5_promptM2k5',
    'chatgpt3p5_promptM3k1', 'chatgpt3p5_promptM3k3', 'chatgpt3p5_promptM3k5',
]


def bold_and_underline(table, metrics_oi, prefix_metric='', inverse_metrics=[]):
    table = table.round(3)
    pm = prefix_metric

    # split the table between rows to skip and those to treat

    for m in metrics_oi:
        # get best and second best
        table[f'{pm}{m}'] = table[f'{pm}{m}'].astype(float)

        if m in inverse_metrics:
            best_metric = table[f'{pm}{m}'].min()
            second_best_metric = table[f'{pm}{m}'].sort_values(ascending=True).unique()[1]
        else:
            best_metric = table[f'{pm}{m}'].max()
            second_best_metric = table[f'{pm}{m}'].sort_values(ascending=False).unique()[1]

        # in bold
        table[f'{pm}{m}'] = table[f'{pm}{m}'].apply(
            lambda x: f'\\textbf{{{float(x):.3f}}}' if str(x) == str(best_metric) else x
        )
        # underlined
        table[f'{pm}{m}'] = table[f'{pm}{m}'].apply(
            lambda x: f'\\underline{{{float(x):.3f}}}' if str(x) == str(second_best_metric)
            else f'{float(x):.3f}' if isinstance(x, float) else x
        )

    return table


def create_table(
        df, refs, datasets_oi, metrics_oi, subset='BM25',
        inverse_metrics=['mean_jaccard_similarity', 'mean_szymkiewicz_similarity', 'mean_bert_similarity']
):
    # reduce precision of metrics to :.2f
    df = df.round(3)

    # select only retriever BM25
    df = df[df['retriever'] == subset]

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
            dsdf = bold_and_underline(dsdf, metrics_oi, inverse_metrics=inverse_metrics)

            for metric in metrics_oi:
                mask = dsdf['reformulation'].eq(ref)
                v = dsdf[mask][metric].values
                v = v[0] if len(v) > 0 else '-'

                row.append(v)
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
\\caption{{Effect that different obfuscations have. The retriever for this table is {subset}.}}
\\label{{tab:table1}}
\\end{{table}}
"""

    lls = 'l' + 'l' * len(datasets_oi) * len(metrics_oi)
    lls_new = 'l' + '|'.join(['c' * len(metrics_oi)] * len(datasets_oi))
    latex = latex.replace(lls, lls_new)

    for ref in refs:
        latex = latex.replace(ref, f'\\textbf{{{ref}}}')

    # cleaning names
    latex = latex.replace('chatgpt3p5_prompt', 'GPTp')
    latex = latex.replace('wordnet', 'WordNet')
    latex = latex.replace('reformulator', '\\textbf{reformulator}')

    metrics_with_at = [m for m in metrics_oi if '@' in m]
    for m in metrics_with_at:
        # latex = latex.replace(m, f'\\makecell{{\\textbf{{{m.split("@")[0]}}} \\\\ \\textbf{{@{m.split("@")[1]}}} }}')
        latex = latex.replace(m, f'\\makecell{{\\textbf{{{m.split("@")[0]}$_{{{m.split("@")[1]}}}$}} }}')

    for m in metrics_oi:
        latex = latex.replace(m, f'\\textbf{{{m}}}')

        if m in inverse_metrics:
            latex = latex.replace(f'{m}}}', f'{m}}}$\downarrow$')

    for i, d in enumerate(datasets_oi):
        d_string = ' & '.join([f'd{i}'] * len(metrics_oi))
        latex = latex.replace(d_string, f'\multicolumn{{{len(metrics_oi)}}}{{c}}{{{d}}} ')

    latex = latex.replace('irds:', '').replace('msmarco-document/', '')
    latex = latex.replace('beir/', '').replace('/test', '')
    latex = latex.replace('trec-dl-2020', 'TREC DL 2020').replace('nfcorpus', 'NFCorpus')
    latex = latex.replace('scifact', 'SciFact').replace('trec-covid', 'TREC-Covid')
    latex = latex.replace('webis-touche2020/v2', "\\textit{Touch\\'e}")

    latex = latex.replace('mean_bert_similarity', '\\textrm{CS}$_{\\textrm{B}}$')
    latex = latex.replace('mean_jaccard_similarity', 'JI')
    latex = latex.replace('mean_szymkiewicz_similarity', 'SSC')
    latex = latex.replace('map', 'MAP')
    latex = latex.replace('reformulator', 'QM')
    latex = latex.replace('none', 'NONE')
    latex = latex.replace('vik\\textbf{', '\\textbf{vik')
    latex = latex.replace('pM1', 'p4-').replace('pM2', 'p5-').replace('pM3', 'p6-')

    for i in [1, 3, 5]:
        latex = latex.replace(f'-k{i}', f'$^{{k={i}}}$')

    for i in range(7):
        latex = latex.replace(f'p{i}', f'$_{{P{i}}}$')

    latex = latex.replace('vikcmp', 'DP V$_{CMP}$').replace('vikm', 'DP V$_M$').replace('vik', 'DP V')
    latex = latex.replace('cmp', 'DP CMP').replace('mah', 'DP M')

    for i in [10, 1, 50, 5]:
        latex = latex.replace(f'_e{i}', f'$_{{{i}}}$')

    latex = latex.replace('$0', '0$').replace('nan', '-').replace('BM25%100>>MonoT5', 'BM25$_{100}\\rightarrow$MonoT5')
    latex = latex.replace('$$', '').replace('}0', '0}').replace('} 0', '0}').replace('}$0', '0}$')

    return latex


# all_latexes = ''
# for dataset in tqdm(all_datasets):
#     print(f"Generating table for {dataset}")
#     latex = create_table(df.copy(), refs, [dataset], metrics_oi_)
#     all_latexes += '\n\n' + latex

datasets_oi = [
    'irds:beir/nfcorpus/test',
    'irds:beir/scifact/test',
    'irds:beir/trec-covid',
    'irds:beir/webis-touche2020/v2',
    # 'irds:msmarco-document/trec-dl-2019',
    # 'irds:msmarco-document/trec-dl-2020',
]

retrievers = df['retriever'].unique()
subset = 'BM25'  # 'BM25' 'BM25%100>>MonoT5'
subset = 'BM25%100>>MonoT5'  # 'BM25' 'BM25%100>>MonoT5'
print(retrievers)

subsets = ['BM25', 'BM25%100>>MonoT5']
# subsets = ['BM25']
for subset in subsets:
    all_latexes = create_table(df.copy(), refs, datasets_oi, metrics_oi_, subset=subset)
    print(all_latexes)

# all_latexes = create_table(df.copy(), refs, ['irds:msmarco-document/trec-dl-2020'], metrics_oi_)
# print(all_latexes)
