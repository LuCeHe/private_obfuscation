import argparse, os, sys, json, time, string, random, shutil

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

from private_obfuscation.helpers_more import do_save_dicts
from private_obfuscation.paths import EXPSDIR

named_tuple = time.localtime()  # get struct_time
time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)

characters = string.ascii_letters + string.digits
random_string = ''.join(random.choice(characters) for i in range(5))
EXPDIR = os.path.join(EXPSDIR, time_string + random_string + '_reformulators')
os.makedirs(EXPDIR, exist_ok=True)

from private_obfuscation.reformulators import reformulate_queries

import pyterrier as pt
# pt.java.init()


# import pyterrier
# from pyterrier import measures
RR, nDCG, P, R, AP = pt.measures.RR, pt.measures.nDCG, pt.measures.P, pt.measures.R, pt.measures.AP
# from pyterrier.measures import RR, nDCG, P, R, AP

from private_obfuscation.retrievers import get_retriever, get_dataset

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
    parser.add_argument("--dataset_name", type=str, default="irds:vaswani")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    return args


def main(args):
    start_time = time.time()
    print(json.dumps(args.__dict__, indent=2))
    results = {}

    # Load a dataset (use any small available dataset)
    index, topics, qrels = get_dataset(args.dataset_name)

    # Reformulate the queries
    topics = reformulate_queries(topics, args.reformulation)

    # Get the retriever
    retrievers = get_retriever(args.dataset_name, args.retriever, index = index)

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


if __name__ == "__main__":
    args = parse_args()
    main(args)
