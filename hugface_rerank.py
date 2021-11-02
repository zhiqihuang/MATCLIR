from hugface_arguments import get_parser
import hugface_train
import hugface_utils
import hugface_data
import torch.nn as nn
import torch
import subprocess
import os

def test_trec_eval(qrelf, runf, metrics):
    trec_eval_f = 'bin/trec_eval' #'-m', metric,
    output = subprocess.check_output([trec_eval_f, qrelf, runf]).decode().rstrip()
    output = output.split('\n')
    eval_out = []
    for line in output:
        linetoks = line.split('\t')
#         print(linetoks[0])
        if linetoks[0].strip() in metrics:
            eval_out.append(line)
            # print(line)
    return eval_out


def main_cli():
    parser = get_parser()
    args = parser.parse_args()

    if args.model_name_or_path == "bert-multilingual-passage-reranking-msmarco":
        args.model_name_or_path = "amberoad/bert-multilingual-passage-reranking-msmarco"

    FunctionMap = {
        "custom_head": {"train": hugface_train.custom_train, "run_model": hugface_train.custom_run_model},
        "custom_placebo_head": {"train": hugface_train.custom_train, "run_model": hugface_train.custom_run_model},
        "vanilla": {"run_model": hugface_train.run_model}
    }

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = hugface_train.RANKER_MODEL_MAP[args.model_ranker](args).to(args.device)
    model.load(args.output_dir + '/weights.p')

    queries, docs = hugface_data.read_datafiles([args.documents, args.queries])  # returns query and documents
    test_run = hugface_data.read_run_dict(args.test_run, topK=args.rerank_topK)

    VALIDATION_METRICS = {'num_q', 'map', 'P_10'}
    runf = os.path.join(args.output_dir, 'test.run')
    if args.model_ranker == "vanilla":
        test_loader = hugface_data.create_vanilla_run_loader(args, model, queries, docs, test_run, 'test', args.per_gpu_eval_batch_size)
    else:
        test_loader = hugface_data.create_custom_run_loader(args, model, queries, docs, test_run, 'test', args.per_gpu_eval_batch_size)


    if torch.cuda.device_count() > 1:
        print("Data Parallel on ", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)
    model.to(args.device)
    if args.tt != None:
        tt = hugface_utils.getTransTableDict(args.tt)
    else:
        tt = None
    FunctionMap[args.model_ranker]['run_model'](args, model, test_loader, runf, 'rerank', tt)
    trec_out = test_trec_eval(args.qrels_dir, runf, VALIDATION_METRICS)

    # writing trec_eval output into a file
    trec_eval_outfile = os.path.join(args.output_dir, 'test-{}.trec_eval'.format(args.rerank_topK))
    bash_file = open(trec_eval_outfile, 'w')
    for line in trec_out:
        bash_file.write(line + '\n')
    bash_file.close()

if __name__ == '__main__':
    main_cli()
