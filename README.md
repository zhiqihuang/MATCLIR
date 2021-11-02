# Mixed Attention Transformer for Leveraging Word-Level Knowledge to Neural Cross-Lingual Information Retrieval

Zhiqi Huang, Hamed Bonab, Sheikh Muhammad Sarwar, Razieh Rahimi, and James Allan

This repo provides the code for reproducing the experiments in [Mixed Attention Transformer for Leveraging Word-Level Knowledge to Neural Cross-Lingual Information Retrieval](https://arxiv.org/abs/2109.02789)

## Requirements

To install requirements, run the following commands:

```
git clone https://github.com/zhiqihuang/MATCLIR.git
cd MATCLIR
pip install -r requirements.txt
```

## Data Download

Later we will upload dataset and translation table to google drive. Stay tuned. You can also run code based on your own data, see training for arguments explanation.

## Training

Note that some of the arguments are not required.

```
python3 hugface_train.py \
        --batches # pre-sampled training batches
        --doc_lang # document langauge
        --documents # document collection
        --fold_num # cross validation fold number
        --freeze_epochs # epochs to freeze translation head parameter 
        --gradient_accumulation_steps # gradient accumulation steps
        --input_dir # data directory
        --learning_rate # learning rate
        --max_non_update_epochs # early stop config
        --model_name_or_path # pretrained model name
        --model_ranker # name of the reranker
        --model_type # model type
        --new_transformer_learning_rate # learning rate for MAT layers 
        --num_train_epochs # max training epoches 
        --output_dir # output directory 
        --overwrite_cache # if overwrite cached data
        --overwrite_output_dir # if overwrite the output directory 
        --per_gpu_eval_batch_size # evaluation batch size
        --per_gpu_train_batch_size # train batch size 
        --qrels_dir # qrel file for evaluation
        --queries # query file
        --query_lang # query langauge
        --rerank_topK # rerank top K from inital retreival 
        --seed 42 # random seed
        --test_run # inital retreival runfile for testing
        --train_pairs # pre-sampled training triplets
        --tt # directory for translation table
        --valid_per_epoch # epochs to run validaiton 
        --valid_run # inital retreival runfile for validation
```

