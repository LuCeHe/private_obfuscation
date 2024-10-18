
import os
import pyterrier as pt
from sentence_transformers import SentenceTransformer
import numpy as np
from pyterrier_colbert.ranking import ColBERTFactory
import pyterrier_colbert.indexing
import pyterrier_colbert as pycolbert
import faiss

from private_obfuscation.paths import DATADIR

# Ensure PyTerrier is started
if not pt.started():
    pt.init()


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


def get_retriever(dataset_name, retriever):
    """
    Returns the specified retriever for the given dataset.

    :param dataset: Dataset name.
    :param retriever: Retriever name.
    :return: PyTerrier retriever object.
    """
    if retriever == "bm25":
        if dataset_name == "vaswani":
            # Initialize BM25 retriever using the dataset index
            retriever =  pt.terrier.Retriever.from_dataset(dataset_name, "terrier_stemmed", wmodel="BM25")
        elif 'trec' in dataset_name.lower():
            # Initialize BM25 retriever using the dataset documents
            documents = pt.get_dataset(dataset_name)
            # retriever =  pt.BatchRetrieve(pt.get_dataset(dataset_name), wmodel="BM25")

            indexer = pt.TRECCollectionIndexer("./index")
            # this downloads the file msmarco-docs.trec.gz
            indexref = indexer.index(documents.get_corpus())
            index = pt.IndexFactory.of(indexref)

            # DPH_br = pt.terrier.Retriever(index, wmodel="DPH") % 100
            retriever = pt.terrier.Retriever(index, wmodel="BM25") % 100




    elif retriever == "faiss":
        raise NotImplementedError("FAISS retriever is not functional yet.")

        # Initialize FAISS retriever using the dataset documents
        documents = dataset.get_corpus()
        retriever =  FaissRetriever(documents)

    elif retriever == "colbert":
        # Initialize ColBERT retriever using the dataset documents
        retriever =  get_colbert_e2e(dataset_name)

    else:
        raise ValueError(f"Invalid retriever: {retriever}")

    return retriever


def tests_with_webis():
    # this code is functional

    import pyterrier as pt
    # dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')
    # indexer = pt.index.IterDictIndexer('./cord19-index')
    # fields = ('title', 'abstract')
    # topics = 'title'
    dataset = pt.datasets.get_dataset('irds:beir/webis-touche2020/v2')
    indexer = pt.index.IterDictIndexer('./webis-index', meta={'docno': 39})
    fields = ('text',)
    topics = 'text'

    indexref = indexer.index(dataset.get_corpus_iter(), fields=fields)
    index = pt.IndexFactory.of(indexref)
    # indexer = pt.TRECCollectionIndexer("./index")
    # indexref = indexer.index(dataset.get_corpus())
    # index = pt.IndexFactory.of(indexref)

    DPH_br = pt.terrier.Retriever(index, wmodel="DPH") % 100
    BM25_br = pt.terrier.Retriever(index, wmodel="BM25") % 100
    # this runs an experiment to obtain results on the TREC COVID queries and qrels
    pt.Experiment(
        [DPH_br, BM25_br],
        dataset.get_topics(topics),
        dataset.get_qrels(),
        eval_metrics=["P.5", "P.10", "ndcg_cut.10", "map"])


def tests_with_covid():
    # this code is functional

    import pyterrier as pt
    # dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')
    # indexer = pt.index.IterDictIndexer('./cord19-index')
    # fields = ('title', 'abstract')
    # topics = 'title'
    dataset = pt.datasets.get_dataset('irds:beir/trec-covid')
    indexer = pt.index.IterDictIndexer('./covid-index', meta={'docno': 39})
    fields = ('text',)
    topics = 'text'

    indexref = indexer.index(dataset.get_corpus_iter(), fields=fields)
    index = pt.IndexFactory.of(indexref)
    # indexer = pt.TRECCollectionIndexer("./index")
    # indexref = indexer.index(dataset.get_corpus())
    # index = pt.IndexFactory.of(indexref)

    DPH_br = pt.terrier.Retriever(index, wmodel="DPH") % 100
    BM25_br = pt.terrier.Retriever(index, wmodel="BM25") % 100
    # this runs an experiment to obtain results on the TREC COVID queries and qrels
    pt.Experiment(
        [DPH_br, BM25_br],
        dataset.get_topics(topics),
        dataset.get_qrels(),
        eval_metrics=["P.5", "P.10", "ndcg_cut.10", "map"])


def tests_with_bair(dataset_name):
    # unsure: irds:beir/quora/test
    nice_ds = ['irds:beir/webis-touche2020/v2', 'irds:beir/trec-covid', 'irds:beir/scifact/train', ]
    assert dataset_name in nice_ds, f"Dataset name must be one of {nice_ds}"
    index_name = f"{dataset_name.replace('irds:', '').split('/')[1]}-index"
    index_path = os.path.join(DATADIR, index_name)

    import pyterrier as pt
    # dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')
    # indexer = pt.index.IterDictIndexer('./cord19-index')
    # fields = ('title', 'abstract')
    # topics = 'title'
    dataset = pt.datasets.get_dataset(dataset_name)
    indexer = pt.index.IterDictIndexer(index_path, meta={'docno': 39})
    fields = ('text',)
    topics = 'text'

    indexref = indexer.index(dataset.get_corpus_iter(), fields=fields)
    index = pt.IndexFactory.of(indexref)
    # indexer = pt.TRECCollectionIndexer("./index")
    # indexref = indexer.index(dataset.get_corpus())
    # index = pt.IndexFactory.of(indexref)

    DPH_br = pt.terrier.Retriever(index, wmodel="DPH") % 100
    BM25_br = pt.terrier.Retriever(index, wmodel="BM25") % 100
    # this runs an experiment to obtain results on the TREC COVID queries and qrels
    pt.Experiment(
        [DPH_br, BM25_br],
        dataset.get_topics(topics),
        dataset.get_qrels(),
        eval_metrics=["P.5", "P.10", "ndcg_cut.10", "map"])
