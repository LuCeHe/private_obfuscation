import pyterrier as pt
from sentence_transformers import SentenceTransformer
import numpy as np
from pyterrier_colbert.ranking import ColBERTFactory
import pyterrier_colbert.indexing
import pyterrier_colbert as pycolbert
import faiss

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