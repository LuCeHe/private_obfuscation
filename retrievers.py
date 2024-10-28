import os
from sentence_transformers import SentenceTransformer

import pyterrier as pt

from private_obfuscation.paths import DATADIR


def get_colbert_e2e(dataset):
    pt.init()

    import pyterrier_colbert.indexing
    import pyterrier_colbert as pycolbert

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
    'irds:msmarco-document/trec-dl-2019/judged',
    'irds:msmarco-document/trec-dl-2020/judged',
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
    fields = ('text',)
    topics_arg = 'text' if not 'trec-dl-' in dataset_name else None

    # unsure: irds:beir/quora/test
    assert dataset_name in nice_ds, f"Dataset name must be one of {nice_ds}"
    dataset_main_name = dataset_name.replace('irds:', '')
    dataset_main_name = dataset_main_name.split('/')[1] if '/' in dataset_main_name else dataset_main_name
    index_name = f"{dataset_main_name}-index"

    print(f"Loading dataset: {dataset_name}")
    dataset = pt.datasets.get_dataset(dataset_name)
    index_path = os.path.join(DATADIR, 'pyterrier-indices', index_name)

    if not os.path.exists(os.path.join(index_path, 'data.properties')):
        print(f"Creating index for dataset: {dataset_name}")

        # if 'msmarco-doc' in dataset_name:
        #     print('here?')
        #     indexer = pt.TRECCollectionIndexer(index_path)
        #     corpus = dataset.get_corpus()
        # else:
        indexer = pt.IterDictIndexer(index_path, meta={'docno': 39}, verbose=True, overwrite=False)
        corpus = dataset.get_corpus_iter()

        print(f"Indexing dataset: {dataset_name}")
        # indexref = indexer.index(dataset.get_corpus_iter(), fields=fields)
        indexref = indexer.index(corpus)
    else:
        if not pt.started():
            pt.init()
        print(f"Loading index for dataset ({dataset_name}): {index_path}")
        indexref = pt.IndexRef.of(os.path.join(index_path, 'data.properties'))

    print(f"Loading index: {index_name}")
    index = pt.IndexFactory.of(indexref)

    print(f"Loading topics and qrels for dataset: {dataset_name}")
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
