# huggingface attempt
# from datasets import load_dataset
#
# queries = load_dataset('irds/trec-fair_2022_train', 'queries', trust_remote_code=True)
#
# print(queries)
# print(queries[0])
#
# qrels = load_dataset('irds/trec-fair_2022_train', 'qrels', trust_remote_code=True)
#
# print(qrels)
# print(qrels[0])

import argparse, random, string

# PyTerrier attempt
import pyterrier as pt

# Initialize PyTerrier
if not pt.started():
    pt.init()

from pyterrier.measures import RR, nDCG, AP

from private_obfuscation.retrievers import FaissRetriever, get_colbert_e2e

obfuscation_types = [
    'none',
    'random_char',
]
retrievers = [
    'bm25',
    'faiss',
    'colbert'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obfuscation", type=str, default="none", choices=obfuscation_types)
    parser.add_argument("--retriever", type=str, default="colbert", choices=retrievers)
    parser.add_argument("--dataset", type=str, default="vaswani")

    args = parser.parse_args()

    return args


# Function to obfuscate queries by replacing random characters
def obfuscate_query(query):
    chars = list(query)
    num_chars_to_change = len(chars) // 3  # Change 1/3 of the characters
    for _ in range(num_chars_to_change):
        idx = random.randint(0, len(chars) - 1)  # Random index
        chars[idx] = random.choice(string.ascii_letters)  # Replace with random letter
    return ''.join(chars)


def main(args):
    # Load a dataset (use any small available dataset)
    dataset = pt.get_dataset(args.dataset)

    # Get the original topics (queries) from the dataset
    topics = dataset.get_topics()

    topics["original_query"] = topics["query"]

    if args.obfuscation == "random_char":
        # Obfuscate the queries
        topics["query"] = topics["query"].apply(obfuscate_query)

    if args.retriever == "bm25":
        # Initialize BM25 retriever using the dataset index
        retriever = pt.terrier.Retriever.from_dataset(dataset, "terrier_stemmed", wmodel="BM25")

    elif args.retriever == "faiss":
        # Initialize FAISS retriever using the dataset documents
        documents = dataset.get_corpus()
        retriever = FaissRetriever(documents)

    elif args.retriever == "colbert":
        # Initialize ColBERT retriever using the dataset documents
        retriever = get_colbert_e2e(args.dataset)

    else:
        raise ValueError(f"Invalid retriever: {args.retriever}")

    # Retrieve results using the obfuscated queries
    results = retriever.transform(topics)

    # Print the obfuscated queries and corresponding results
    i = 0
    for index, row in topics.iterrows():
        print("-" * 50)
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
    print(results)
    print(experiment.to_string())


if __name__ == "__main__":
    args = parse_args()
    main(args)
