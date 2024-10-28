import argparse, os, sys, json, time, string, random, shutil

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

import pyterrier as pt

if not pt.started():
    pt.init()

RR, nDCG, P, R, AP = pt.measures.RR, pt.measures.nDCG, pt.measures.P, pt.measures.R, pt.measures.AP

from private_obfuscation.retrievers import get_retriever, get_dataset
from private_obfuscation.helpers_more import do_save_dicts, create_exp_dir
from private_obfuscation.paths import PODATADIR
from private_obfuscation.helpers_llms import refs_types, dp_refs
from private_obfuscation.reformulators import reformulate_queries

reformulation_types = [
    'none',
    # 'random_char',
    # 'chatgpt_improve',
    *[f'chatgpt3p5_{rt}' for rt in list(refs_types.keys())],
    *dp_refs,
    'wordnet',
]
retrievers = [
    'bm25',
    'colbert',
    'monoT5',
]

ds = [
    # 'irds:vaswani',
    'irds:beir/nfcorpus/test',
    'irds:beir/scifact/test',
    'irds:beir/trec-covid',
    'irds:beir/webis-touche2020/v2',
    # 'irds:beir/arguana',
    'irds:msmarco-document/trec-dl-2019',
    'irds:msmarco-document/trec-dl-2020',
    # 'irds:msmarco-document/trec-dl-2020/judged',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reformulation", type=str, default="wordnet", choices=reformulation_types)
    parser.add_argument("--retriever", type=str, default="monoT5", choices=retrievers)
    parser.add_argument("--dataset_name", type=str, default="irds:beir/nfcorpus/test")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    return args


def main(args):
    EXPDIR = create_exp_dir()
    start_time = time.time()
    print(json.dumps(args.__dict__, indent=2))
    results = {}

    # Load a dataset (use any small available dataset)
    topics, qrels, index, indexref, dataset = get_dataset(args.dataset_name)

    print('Reformulating queries...')
    topics = reformulate_queries(topics, args.reformulation, dataset_name=args.dataset_name)

    print('Getting retriever...')
    retrievers = get_retriever(args.dataset_name, args.retriever, index=index, indexref=indexref, dataset=dataset)

    # Optionally, evaluate the results if you have qrels (ground truth)
    experiment = pt.Experiment(
        retrievers, topics, qrels,
        eval_metrics=[
            "map", "recip_rank",
            RR(rel=2), RR @ 10, RR @ 100,
                       nDCG @ 10, nDCG @ 100,
                       P @ 10, P @ 100, P @ 1000,
                       R @ 50, R @ 1000,
            AP(rel=2)
        ])
    # show output experiments
    print(experiment.to_string())

    # experiment to json
    experiment_json = experiment.to_dict()
    results.update(experiment_json)

    print(json.dumps(results, indent=2))
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time
    print('Elapsed time', elapsed_time)

    save_dicts = {"args": args.__dict__, "results": results}
    do_save_dicts(save_dicts, save_dir=EXPDIR)
    shutil.make_archive(EXPDIR, 'zip', EXPDIR)
    print('Done')


def loop_all_over_reformulations():
    # save the args of the experiments already run, so I don't run them again
    done_experiments = []
    path = os.path.join(PODATADIR, 'done_experiments.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            done_experiments = json.load(f)
    else:
        with open(path, 'w') as f:
            json.dump(done_experiments, f)

    retrivs = ['monoT5']
    i = 0
    for dataset_name in ds:
        for reformulation in reformulation_types:
            for retriever in retrivs:
                i += 1
                print(f'{i}/{len(ds) * len(reformulation_types) * len(retrivs)}')

                if any([d['reformulation'] == reformulation and d['retriever'] == retriever and d[
                    'dataset_name'] == dataset_name for d in done_experiments]):
                    print('Already done')
                    continue

                try:
                    # if True:
                    args = argparse.Namespace(
                        reformulation=reformulation, retriever=retriever, dataset_name=dataset_name
                    )
                    main(args)

                    print('Saving experiment as done')
                    done_experiments.append(args.__dict__)
                    with open(path, 'w') as f:
                        json.dump(done_experiments, f)

                except Exception as e:
                    print('Error:', e)
                    continue


if __name__ == "__main__":
    args = parse_args()

    if args.notes == 'loop':
        loop_all_over_reformulations()
    else:
        main(args)
