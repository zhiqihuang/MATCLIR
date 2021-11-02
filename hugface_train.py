import os
import sys
import subprocess
import shutil
import random
import logging
from torch.utils.data import dataloader
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from hugface_arguments import get_parser
import hugface_data
import hugface_models
import hugface_utils

RANKER_MODEL_MAP = {
    'custom_head': hugface_models.CustomTransformerRankerTranslationHead,
    'custom_placebo_head': hugface_models.CustomTransformerRankerPlaceboHead,
    'vanilla': hugface_models.VanillaTransformerRanker,
}

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

def setup_myloger(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    return logger

def train(args, model, optimizer, data_loader, tt, epoch):
    steps = 0
    total_loss = 0.0
    model.train()

    for record in tqdm(data_loader, desc="training-{}".format(epoch)):

        scores = model(record['query_tok'],
                        record['query_mask'],
                        record['doc_tok'],
                        record['doc_mask'])
        steps += 1
        count = len(record['query_id']) // 2
        scores = scores.reshape(count, 2)
        loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pair_wise softmax
        loss.backward()
        total_loss += loss.item()
        if steps % args.gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
    return total_loss

def custom_train(args, model, optimizer, data_loader, tt, epoch):
    steps = 0
    total_loss = 0.0
    model.train()

    for record in tqdm(data_loader, desc="training-{}".format(epoch)):
        
        scores = model(record['query_tok'],
                        record['query_mask'],
                        record['doc_tok'],
                        record['doc_mask'],
                        record['query_sub_index'],
                        record['doc_sub_index'],
                        record['query_words_txt'],
                        record['doc_words_txt'],
                        tt,
                        args.tt_threshold,
                        args.normalization,
                        args.device)

        steps += 1
        count = len(record['query_id']) // 2
        scores = scores.reshape(count, 2)
        loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pair_wise softmax
        loss.backward()
        total_loss += loss.item()
        if steps % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    return total_loss

def validate(args, fun_map, model, data_loader, tt, epoch):
    VALIDATION_METRIC = 'map'
    qrels_f = args.qrels_dir
    runf = os.path.join(args.output_dir, f'{epoch}.run')
    fun_map[args.model_ranker]['run_model'](args, model, data_loader, runf, epoch, tt)
    return trec_eval(qrels_f, runf, VALIDATION_METRIC)

def run_model(args, model, data_loader, runf, epoch, tt=None):
    rerank_run = {}
    with torch.no_grad():
        model.eval()
        for records in tqdm(data_loader, desc="{}-{}".format(data_loader.dataset.name, epoch)):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()

    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def custom_run_model(args, model, data_loader, runf, epoch, tt):
    rerank_run = {}
    with torch.no_grad():
        model.eval()
        for records in tqdm(data_loader, desc="{}-{}".format(data_loader.dataset.name, epoch)):
            scores = model(records['query_tok'],
                            records['query_mask'],
                            records['doc_tok'],
                            records['doc_mask'],
                            records['query_sub_index'], 
                            records['doc_sub_index'],  
                            records['query_words_txt'],
                            records['doc_words_txt'],
                            tt, 
                            args.tt_threshold,
                            args.normalization,
                            args.device)
            
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()

    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')
         
def trec_eval(qrelf, runf, metric):
    trec_eval_f = 'bin/trec_eval'
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.model_name_or_path == "bert-multilingual-passage-reranking-msmarco":
        args.model_name_or_path = "amberoad/bert-multilingual-passage-reranking-msmarco"

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if not torch.cuda.is_available():
        raise ValueError("No cuda available, exit!")

    # Setup logging & Set seed
    logger = setup_myloger(args)
    set_seed(args)

    # input and output handling
    queries, docs = hugface_data.read_datafiles([args.documents, args.queries])  # returns query and documents
    print("queries", len(queries), "docs", len(docs))
    qrels = hugface_data.read_qrels_dict(args.qrels_dir)  # returns qrels
    print("qrels", len(qrels))
    train_pairs = hugface_data.read_pairs_dict(args.train_pairs)  # returns training pairs
    print("train_pairs", len(train_pairs))
    valid_run = hugface_data.read_run_dict(args.valid_run, topK=args.rerank_topK) # returns validation pairs
    print("valid_run", len(valid_run))
    batches = hugface_data.read_batches(args.batches)
    
    #output dir
    os.makedirs(args.output_dir, exist_ok=True)
    if args.overwrite_output_dir:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    #load translation table
    if args.tt != None:
        tt = hugface_utils.getTransTableDict(args.tt)
    else:
        tt = None

    # model
    model = RANKER_MODEL_MAP[args.model_ranker](args).to(args.device)
    
    if torch.cuda.device_count() > 1:
        print("Data Parallel on ", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)
    model.to(args.device)

    if args.continue_training:
        model.load(args.continue_training)
    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    cls_params = {'params': [v for k, v in params if k.startswith('cls.')]}
    if args.model_type == "bert":
        new_transformer_params = {'params': [v for k, v in params if (k.startswith('transEncoder.') or k.startswith('fixedEncoder.'))],
                    'lr': args.new_transformer_learning_rate}
        transformer_params = {'params': [v for k, v in params if k.startswith('transformer.')],
                    'lr': args.transformer_learning_rate}
        optimizer = torch.optim.Adam([cls_params, new_transformer_params, transformer_params], lr=args.learning_rate)
    
    elif args.model_type == "bert_internal":
        new_transformer_params = {'params': [v for k, v in params if k.startswith('transformer.encoder.fixedEncoder')],
                    'lr': args.new_transformer_learning_rate}
        transformer_params = {'params': [v for k, v in params if (k.startswith('transformer.') and not k.startswith('transformer.encoder.fixedEncoder'))],
                    'lr': args.transformer_learning_rate}
        optimizer = torch.optim.Adam([cls_params, new_transformer_params, transformer_params], lr=args.learning_rate)
    
    else:
        transformer_params = {'params': [v for k, v in params if k.startswith('transformer.')],
                    'lr': args.transformer_learning_rate}
        optimizer = torch.optim.Adam([cls_params, transformer_params], lr=args.learning_rate)


    num_param = count_parameters(model)
    print("Model has number of parameters ", num_param)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    if args.model_ranker == "vanilla":
        train_loader = hugface_data.create_vanilla_train_loader(args, model, queries, docs, train_pairs, qrels, batches, args.train_batch_size)
        valid_loader = hugface_data.create_vanilla_run_loader(args, model, queries, docs, valid_run, 'valid', args.eval_batch_size)
    else:
        train_loader = hugface_data.create_custom_train_loader(args, model, queries, docs, train_pairs, qrels, batches, args.train_batch_size)
        valid_loader = hugface_data.create_custom_run_loader(args, model, queries, docs, valid_run, 'valid', args.eval_batch_size)
    
    FunctionMap = {
        "custom_head": {"train": custom_train, "run_model": custom_run_model},
        "custom_placebo_head": {"train": custom_train, "run_model": custom_run_model},
        "vanilla": {"train": train, "run_model": run_model}
    }
    
    top_valid_score = 0
    top_valid_score_epoch = 0

    # freeze pretrained model
    if args.freeze_epochs > 0:
        print(f'freeze pretrained model for {args.freeze_epochs} epochs.')
        if args.model_type == "bert_internal":
            if hasattr(model, "module"):
                for k, v in model.module.named_parameters():
                    if k.startswith('transformer.') and not k.startswith('transformer.encoder.fixedEncoder'):
                        v.requires_grad = False
            else:
                for k, v in model.named_parameters():
                    if k.startswith('transformer.') and not k.startswith('transformer.encoder.fixedEncoder'):
                        v.requires_grad = False
        else:
            if hasattr(model, "module"):
                for k, v in model.module.named_parameters():
                    if k.startswith('transformer.'):
                        v.requires_grad = False
            else:
                for k, v in model.named_parameters():
                    if k.startswith('transformer.'):
                        v.requires_grad = False
    

    for epoch in range(args.num_train_epochs):
        train_loader.dataset.epoch = epoch + 1
        train_loader.sampler.epoch = epoch + 1

        # unfreeze
        if args.freeze_epochs > 0 and epoch + 1 == args.freeze_epochs:
            print(f'unfreeze pretrained model.')
            if hasattr(model, "module"):
                for k, v in model.module.named_parameters():
                    v.requires_grad = True
            else:
                for k, v in model.named_parameters():
                    v.requires_grad = True
            top_valid_score_epoch = epoch # count top_valid_score after unfreezing

        loss = FunctionMap[args.model_ranker]['train'](args, model, optimizer, train_loader, tt, epoch)
        print(f'train epoch={epoch} loss={loss}')
        
        if (epoch+1) % args.valid_per_epoch == 0:
            valid_score = validate(args, FunctionMap, model, valid_loader, tt, epoch)
            print(f'validation epoch={epoch} score={valid_score}')
            if valid_score > top_valid_score:
                top_valid_score = valid_score
                top_valid_score_epoch = epoch
                print('new top validation score, saving weights')
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save(os.path.join(args.output_dir, 'weights.p'))
        
        if epoch - top_valid_score_epoch > args.max_non_update_epochs:
            break
        
        # zhiqi: check results after each epoch
        sys.stdout.flush()

