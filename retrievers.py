import os
from sentence_transformers import SentenceTransformer

import pyterrier as pt

pt.init()

import pyterrier_colbert.indexing
import pyterrier_colbert as pycolbert
# import faiss

from private_obfuscation.paths import DATADIR


class FaissRetriever(pt.Transformer):
    def __init__(self, documents, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the FAISS retriever.

        :param documents: List of documents (strings) to index.
        :param model_name: Transformer model for encoding documents and queries.
        """
        # Load transformer model for embeddings
        self.model = SentenceTransformer(model_name)

        # Convert documents to embeddings
        self.documents = documents
        self.document_embeddings = self.model.encode(documents, convert_to_tensor=False).astype('float32')

        # Create FAISS index for nearest neighbor search
        self.index = faiss.IndexFlatL2(self.document_embeddings.shape[1])  # L2 distance
        self.index.add(self.document_embeddings)

    def transform(self, queries):
        """
        Transforms the input queries into results using FAISS.

        :param queries: Pandas DataFrame with a 'query' column.
        :return: DataFrame with columns 'qid', 'docno', 'rank', 'score', and 'docid'.
        """
        # Convert queries into dense embeddings
        query_embeddings = self.model.encode(queries["query"].tolist(), convert_to_tensor=False).astype('float32')

        # Number of top results to return
        k = 100  # You can adjust this depending on how many results you need

        # Search the FAISS index
        distances, indices = self.index.search(query_embeddings, k)

        results = []
        for i, qid in enumerate(queries["qid"]):
            for rank, (doc_idx, score) in enumerate(zip(indices[i], distances[i])):
                results.append({
                    'qid': qid,
                    'docno': str(doc_idx),  # Document ID
                    'rank': rank + 1,
                    'score': -score,  # Negative L2 distance to turn it into a "score"
                    'docid': doc_idx  # This is an internal document ID; map it if needed
                })

        # Convert results into a DataFrame
        return pt.Dataframe(results)


def get_colbert_e2e(dataset):
    """
    Returns the ColBERT end-to-end retrieval model.

    :return: PyTerrier transformer object.
    """
    checkpoint = "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"

    indexer = pycolbert.indexing.ColBERTIndexer(checkpoint, "/content", "colbertindex", chunksize=3)
    indexer.index(pt.get_dataset('irds:' + dataset).get_corpus_iter())

    pyterrier_colbert_factory = indexer.ranking_factory()

    colbert_e2e = pyterrier_colbert_factory.end_to_end()
    return colbert_e2e


def get_retriever(dataset_name, retriever, index=None):
    """
    Returns the specified retriever for the given dataset.

    :param dataset: Dataset name.
    :param retriever: Retriever name.
    :return: PyTerrier retriever object.
    """
    retrievers = []
    if retriever == "bm25":
        DPH_br = pt.terrier.Retriever(index, wmodel="DPH") % 100
        DPH = pt.terrier.Retriever(index, wmodel="DPH")
        BM25_br = pt.terrier.Retriever(index, wmodel="BM25") % 100
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        retrievers = [DPH_br, DPH, BM25_br, BM25]

    elif retriever == "faiss":
        raise NotImplementedError("FAISS retriever is not functional yet.")

    elif retriever == "colbert":
        # Initialize ColBERT retriever using the dataset documents
        colbert = get_colbert_e2e(dataset_name)

        DPH_br = pt.terrier.Retriever(index, wmodel="DPH") % 100
        DPH = pt.terrier.Retriever(index, wmodel="DPH")
        BM25_br = pt.terrier.Retriever(index, wmodel="BM25") % 100
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        retrievers = [DPH_br, DPH, BM25_br, BM25, colbert]

    else:
        raise ValueError(f"Invalid retriever: {retriever}")

    return retrievers


bigger_ds = [
    'irds:beir/dbpedia-entity/dev', 'irds:beir/dbpedia-entity/test',
    'irds:msmarco-document/trec-dl-2019',
    'irds:msmarco-document/trec-dl-2020',
    'irds:msmarco-passage/eval',
    'irds:beir/fever/dev', 'irds:beir/fever/test',
]
nice_ds = [
    'irds:vaswani', 'irds:beir/arguana',
    'irds:beir/scifact/dev', 'irds:beir/scifact/test',
    'irds:beir/webis-touche2020/v2',
    'irds:beir/nfcorpus/dev', 'irds:beir/nfcorpus/test',
    'irds:beir/trec-covid',
    'irds:beir/nq',
]
nice_ds += bigger_ds


def get_dataset(dataset_name):
    # unsure: irds:beir/quora/test
    assert dataset_name in nice_ds, f"Dataset name must be one of {nice_ds}"
    dataset_main_name = dataset_name.replace('irds:', '')
    dataset_main_name = dataset_main_name.split('/')[1] if '/' in dataset_main_name else dataset_main_name
    index_name = f"{dataset_main_name}-index"
    index_path = os.path.join(DATADIR, index_name)

    dataset = pt.datasets.get_dataset(dataset_name)
    indexer = pt.index.IterDictIndexer(index_path, meta={'docno': 39}, verbose=True, overwrite=True)
    fields = ('text',)
    topics_arg = 'text'

    indexref = indexer.index(dataset.get_corpus_iter(), fields=fields)
    index = pt.IndexFactory.of(indexref)

    topics = dataset.get_topics(topics_arg)
    qrels = dataset.get_qrels()
    return index, topics, qrels


def tests_with_bair(dataset_name):
    index, topics, qrels = get_dataset(dataset_name)
    print(topics.head())

    DPH_br = pt.terrier.Retriever(index, wmodel="DPH") % 100
    BM25_br = pt.terrier.Retriever(index, wmodel="BM25") % 100
    BM25 = pt.terrier.Retriever(index, wmodel="BM25")
    # this runs an experiment to obtain results on the TREC COVID queries and qrels
    experiment = pt.Experiment(
        [DPH_br, BM25_br, BM25],
        topics,
        qrels,
        eval_metrics=["P.5", "P.10", "ndcg_cut.10", "map"])
    print(experiment.to_string())


if __name__ == '__main__':
    # tests_with_webis()
    # tests_with_covid()
    tests_with_bair('irds:beir/dbpedia-entity/dev')
    # tests_with_bair('irds:beir/trec-covid')
    # tests_with_bair('irds:beir/scifact/dev')
    # tests_with_bair('irds:beir/nq')
    # tests_with_bair('irds:beir/nfcorpus/dev')
    pass
