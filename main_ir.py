import argparse, os, sys, json

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

from private_obfuscation.reformulators import obfuscate_queries

import pyterrier as pt

# Initialize PyTerrier
if not pt.started():
    # pt.init(version='snapshot')
    pt.init()

from pyterrier.measures import RR, nDCG, AP

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reformulation", type=str, default="none", choices=reformulation_types)
    parser.add_argument("--retriever", type=str, default="bm25", choices=retrievers)
    parser.add_argument("--dataset", type=str, default="vaswani")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    return args


def main(args):
    print(json.dumps(args.__dict__, indent=2))

    # Load a dataset (use any small available dataset)
    dataset = pt.get_dataset(args.dataset)

    # Get the original topics (queries) from the dataset
    topics = dataset.get_topics()

    # Reformulate the queries
    topics = reformulate_queries(topics, args.reformulation)

    retriever = get_retriever(args.dataset, args.retriever)

    # Retrieve results using the obfuscated queries
    results = retriever.transform(topics)

    if 'printobfres' in args.notes:
        # Print the obfuscated queries and corresponding results
        i = 0
        for index, row in topics.iterrows():
            print("-" * 50)
            print(f"Query {i + 1}")
            query = row['query']
            print(f"Original Query: {row['original_query']}")
            print(f"Obfuscated Query: {query}")
            print("Top Results:")
            print(results[results['query'] == query][['docno', 'rank', 'score']].head(), '\n')
            i += 1
            if i == 5:
                break

    # Optionally, evaluate the results if you have qrels (ground truth)
    qrels = dataset.get_qrels()
    experiment = pt.Experiment([retriever], topics, qrels,
                               eval_metrics=["map", "recip_rank", RR(rel=2), nDCG @ 10, nDCG @ 100, AP(rel=2)])

    # show output experiments
    print(experiment.to_string())


if __name__ == "__main__":
    args = parse_args()
    main(args)
