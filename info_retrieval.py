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


# PyTerrier attempt
import pyterrier as pt
from pyterrier.measures import *
import random
import string

# Initialize PyTerrier
if not pt.started():
    pt.init()

# Load a dataset (use any small available dataset)
dataset = pt.get_dataset("vaswani")

# Function to obfuscate queries by replacing random characters
def obfuscate_query(query):
    chars = list(query)
    num_chars_to_change = len(chars) // 3  # Change 1/3 of the characters
    for _ in range(num_chars_to_change):
        idx = random.randint(0, len(chars) - 1)  # Random index
        chars[idx] = random.choice(string.ascii_letters)  # Replace with random letter
    return ''.join(chars)

# Get the original topics (queries) from the dataset
topics = dataset.get_topics()

# Obfuscate the queries
topics["query"] = topics["query"].apply(obfuscate_query)

# Initialize BM25 retriever using the dataset index
bm25 = pt.terrier.Retriever.from_dataset(dataset, "terrier_stemmed", wmodel="BM25")

# Retrieve results using the obfuscated queries
results = bm25.transform(topics)

# Print the obfuscated queries and corresponding results
for index, row in topics.iterrows():
    query = row['query']
    print(f"Obfuscated Query: {query}")
    print("Top Results:")
    print(results[results['query'] == query][['docno', 'rank', 'score']].head(), '\n')

# Optionally, evaluate the results if you have qrels (ground truth)
qrels = dataset.get_qrels()
experiment = pt.Experiment([bm25], topics, qrels, eval_metrics=["map", "recip_rank", RR(rel=2), nDCG@10, nDCG@100, AP(rel=2)])


# show output experiments
print(results)
print(experiment.to_string())
