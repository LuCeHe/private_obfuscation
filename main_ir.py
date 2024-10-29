import argparse, os, sys, json, time, shutil

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

import pyterrier as pt

if not pt.started():
    pt.init()

RR, nDCG, P, R, AP = pt.measures.RR, pt.measures.nDCG, pt.measures.P, pt.measures.R, pt.measures.AP

from private_obfuscation.retrievers import get_retriever, get_dataset
from private_obfuscation.helpers_more import do_save_dicts, create_exp_dir
from private_obfuscation.paths import PODATADIR, LOCAL_DATADIR
from private_obfuscation.helpers_more import all_ds, all_reformulation_types, \
    all_retrievers
from private_obfuscation.reformulators import reformulate_queries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reformulation", type=str, default="wordnet", choices=all_reformulation_types)
    parser.add_argument("--retriever", type=str, default="bm25", choices=all_retrievers)
    parser.add_argument("--dataset_name", type=str, default="irds:msmarco-document/trec-dl-2020", choices=all_ds)
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


def loop_all_over_reformulations(notes):
    import socket
    hostserver = socket.gethostname()
    # save the args of the experiments already run, so I don't run them again
    missing_experiments = []
    path = os.path.join(LOCAL_DATADIR, 'missing_experiments.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            missing_experiments = json.load(f)
    else:
        with open(path, 'w') as f:
            json.dump(missing_experiments, f)

    retrivs = ['monoT5']
    # retrivs = ['bm25']
    # all_ds = ['irds:msmarco-document/trec-dl-2020']
    i = 0
    missing_i = 0

    all_ds_ = all_ds
    all_reformulation_types_ = all_reformulation_types
    retrivs_ = retrivs

    if 'reversed' in notes:
        all_ds_ = all_ds[::-1]
        all_reformulation_types_ = all_reformulation_types[::-1]
        retrivs_ = retrivs[::-1]

    for dataset_name in all_ds_:
        for reformulation in all_reformulation_types_:
            for retriever in retrivs_:
                i += 1
                print(f'{i}/{len(all_ds) * len(all_reformulation_types) * len(retrivs)}')

                # # if running in beluga do only even
                # if hostserver.startswith('bg') and i % 2 != 0:
                #     continue
                #
                # # if in narval the opposite
                # if hostserver.startswith('nr') and i % 2 == 0:
                #     continue

                if not any([
                    d['reformulation'] == reformulation
                    and d['retriever'] == retriever
                    and d['dataset_name'] == dataset_name
                    for d in missing_experiments[retriever]
                ]):
                    print('Already done')
                    continue

                missing_i += 1
                try:
                    # if True:
                    args = argparse.Namespace(
                        reformulation=reformulation, retriever=retriever, dataset_name=dataset_name
                    )
                    main(args)

                    # print('Saving experiment as done')
                    # done_experiments.append(args.__dict__)
                    # with open(path, 'w') as f:
                    #     json.dump(done_experiments, f)

                except Exception as e:
                    print('Error:', e)
                    continue

    # print(f'Number of missing experiments: {missing_i}/{len(all_ds) * len(all_reformulation_types) * len(retrivs)}')


if __name__ == "__main__":
    args = parse_args()

    if 'loop' in args.notes:
        loop_all_over_reformulations(notes=args.notes)
    else:
        main(args)
