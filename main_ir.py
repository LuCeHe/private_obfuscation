import argparse, os, sys, json

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

from private_obfuscation.reformulators import reformulate_queries

import pyterrier as pt

# Initialize PyTerrier
if not pt.started():
    # pt.init(version='snapshot')
    pt.init()

from pyterrier.measures import *

from private_obfuscation.retrievers import get_retriever

reformulation_types = [
    'none',
    'random_char',
    'chatgpt_improve',
]
retrievers = [
    'bm25',
    'colbert'
]

datasets_tested = [
    'vaswani',
    'trec-robust-2004'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reformulation", type=str, default="none", choices=reformulation_types)
    parser.add_argument("--retriever", type=str, default="bm25", choices=retrievers)
    parser.add_argument("--dataset_name", type=str, default="vaswani")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    return args


def main(args):
    print(json.dumps(args.__dict__, indent=2))

    # Load a dataset (use any small available dataset)
    dataset = pt.get_dataset(args.dataset_name)

    # Get the original topics (queries) from the dataset
    topics = dataset.get_topics()

    # Reformulate the queries
    topics = reformulate_queries(topics, args.reformulation)

    # Get the retriever
    retriever = get_retriever(args.dataset_name, args.retriever)

    # Optionally, evaluate the results if you have qrels (ground truth)
    qrels = dataset.get_qrels()
    experiment = pt.Experiment(
        [retriever], topics, qrels,
        eval_metrics=[
            "map", "recip_rank", RR(rel=2), RR @ 10, RR @ 100,
            nDCG @ 10, nDCG @ 100,
            P @ 10, P @ 100, P @ 1000,
            R @50, R @ 1000,
            AP(rel=2)
        ])

    # show output experiments
    print(experiment.to_string())


if __name__ == "__main__":
    args = parse_args()
    main(args)
