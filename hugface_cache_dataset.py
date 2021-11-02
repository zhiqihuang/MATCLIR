import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np

from hugface_arguments import get_parser
import hugface_data
import hugface_models

# utils misc.
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser()

    # Required parameters (data)
    parser.add_argument(
        "--documents",
        default=None,
        type=str,
        required=True,
        help="The corpus dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--input_dir", default="", type=str, help="input data diretory",
    )
    parser.add_argument(
        "--total_folds",
        default=5,
        type=int,
        help="total number of folds",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list"
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--sampler", type=str, default=None, help="how to sample pos and neg examples")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--rerank_topK", default=10000, type=int, help="rerank top k docs",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation datasets"
    )
    parser.add_argument(
        "--query_lang",
        default="",
        type=str,
        help="query lang",
    )
    parser.add_argument(
        "--doc_lang",
        default="",
        type=str,
        help="document lang",
    )
    parser.add_argument("--fixed_layer", type=int, default=-1, help="position to insert the fixed atten layer")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.model_name_or_path == "bert-multilingual-passage-reranking-msmarco":
        args.model_name_or_path = "amberoad/bert-multilingual-passage-reranking-msmarco"

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if not torch.cuda.is_available():
        raise ValueError("No cuda available, exit!")

    # Set seed
    set_seed(args)

    args.model_type = "bert" # doesn't matter, we need tokenizer here
    args.model_ranker = "" # doesn't matter, not running rankers

    # model
    model = hugface_models.TransformerRanker(args).to(args.device)
    if torch.cuda.device_count() > 1:
        print("Data Parallel on ", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)
    model.to(args.device)

    args.queries = os.path.join(args.input_dir, "queries.tsv")
    args.qrels_dir = os.path.join(args.input_dir, "qrels")
    queries, docs = hugface_data.read_datafiles([args.documents, args.queries])  # returns query and documents
    print("queries", len(queries), "docs", len(docs))
    qrels = hugface_data.read_qrels_dict(args.qrels_dir)  # returns qrels
    print("qrels", len(qrels))

    for fold_num in range(1, args.total_folds+1):
        args.fold_num = fold_num

        print("caching fold num {}".format(args.fold_num))

        args.train_pairs = os.path.join(args.input_dir, "f{}.train.pairs".format(fold_num))
        train_pairs = hugface_data.read_pairs_dict(args.train_pairs)  # returns training pairs
        print("train_pairs", len(train_pairs))

        args.valid_run = os.path.join(args.input_dir, "f{}.valid.run".format(fold_num))
        valid_run = hugface_data.read_run_dict(args.valid_run, args.rerank_topK) # returns validation pairs
        print("valid_run", len(valid_run))

        args.test_run = os.path.join(args.input_dir, "f{}.test.run".format(fold_num))
        test_run = hugface_data.read_run_dict(args.test_run, args.rerank_topK)
        print("test_run", len(valid_run))

        args.batches = os.path.join(args.input_dir, "f{}.batches.json".format(fold_num))
        batches = hugface_data.read_batches(args.batches)

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        train_loader = hugface_data.create_vanilla_train_loader(args, model, queries, docs, train_pairs, qrels, batches, args.train_batch_size)
        valid_loader = hugface_data.create_vanilla_run_loader(args, model, queries, docs, valid_run, 'valid', args.eval_batch_size)
        test_loader = hugface_data.create_vanilla_run_loader(args, model, queries, docs, test_run, 'test', args.per_gpu_eval_batch_size)

        train_loader = hugface_data.create_custom_train_loader(args, model, queries, docs, train_pairs, qrels, batches, args.train_batch_size)
        valid_loader = hugface_data.create_custom_run_loader(args, model, queries, docs, valid_run, 'valid', args.eval_batch_size)
        test_loader = hugface_data.create_custom_run_loader(args, model, queries, docs, test_run, 'test', args.per_gpu_eval_batch_size)