import argparse
from hugface_models import hugface_models

def get_parser():
    """
    Generate a parameters parser.
    """
    MODEL_CLASSES = hugface_models()

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
        "--queries",
        default=None,
        type=str,
        required=True,
        help="The query text dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--qrels_dir",
        default=None,
        type=str,
        required=True,
        help="The query relevant judgments in TREC style. ",
    )

    parser.add_argument(
        "--train_pairs",
        default=None,
        type=str,
        required=True,
        help="Training data file.",
    )
    parser.add_argument(
        "--valid_run",
        default=None,
        type=str,
        required=True,
        help="Validation data file.",
    )

    # Required parameters (pre-trained model)
    parser.add_argument(
        "--model_ranker",
        default='vanilla',
        type=str,
        required=True,
        help="The model architecture used for ranking like: Vanilla, DRMM, KNRM..."
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    #  rerank arguments (for testing)
    parser.add_argument(
        "--test_run",
        default=None,
        type=str,
        required=False,
        help="Test data file.",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--sampler", type=str, default=None, help="how to sample pos and neg examples")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--transformer_learning_rate", default=5e-5, type=float, help="The initial learning rate for Bert-like transformer.")
    parser.add_argument("--new_transformer_learning_rate", default=5e-5, type=float, help="The initial learning rate for additional transformer layers")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")


    parser.add_argument("--per_epoch_num_batches", default=8, type=int,
        help="For each query many possibilities of selecting pos/neg documents exist, so we bind this with num batches per epoch",
    )
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")



    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
                        )


    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--continue_training",
        default=None,
        type=str,
        help="load previous weights to the model and train on new data",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
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
    parser.add_argument("--valid_per_epoch", type=int, default=1, help="epochs between two valid runs")
    parser.add_argument("--max_non_update_epochs", type=int, default=10, help="max non update epochs, for early stop")
    parser.add_argument("--tt_threshold", type=float, default=0.0, help="threshold for translation table")
    parser.add_argument(
        "--ttmat_reduction", default="norm", type=str, help="how to normalize tt matrix",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--rerank_name", default="", type=str, help="rerank output file name",
    )

    parser.add_argument(
        "--rerank_topK", default=10000, type=int, help="rerank top k docs",
    )
    parser.add_argument(
        "--fold_num", default=1, type=int, help="fold number",
    )
    parser.add_argument(
        "--input_dir", default="", type=str, help="input data diretory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation datasets"
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--tt",
        default=None,
        type=str,
        help="translation table",
    )
    parser.add_argument(
        "--batches",
        default=None,
        type=str,
        help="predefined train batches",
    )
    parser.add_argument(
        "--normalization",
        default="norm",
        type=str,
        help="how to norm ttmat",
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
    parser.add_argument("--freeze_epochs", default=0, type=int, help="epochs to freeze pretrained transformer model")
    return parser



# converting parameters in CEDR to these parameters
# LR = args.learning_rate
# BERT_LR = args.transformer_learning_rate
# MAX_EPOCH = args.num_train_epochs
# BATCH_SIZE = args.per_gpu_train_batch_size
# BATCHES_PER_EPOCH = args.per_epoch_num_batches
# GRAD_ACC_SIZE = arg.gradient_accumulation_steps