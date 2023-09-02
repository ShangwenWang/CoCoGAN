import logging
import multiprocessing
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_pretrain", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--save_result", action='store_true',
                        help="Whether to save jsonl result.")
    parser.add_argument("--do_train_generator", action='store_true',
                        help="Whether to train generator")
    parser.add_argument("--do_train_discriminator", action='store_true',
                        help="Whether to train discriminator")
    parser.add_argument('--hpc', action='store_true', default=False,
                        help='set to hpc mode')
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--similarity_filename", default=None, type=str,
                        help="The similarity rankings of codes in the training set.")
    parser.add_argument("--language", default="python", type=str,
                        help="The used programming language in training corpus")
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--rounds', type=int, default=30, metavar='N',
                        help='rounds of adversarial training (default: 150)')
    parser.add_argument('--g_pretrain_steps', type=int, default=120, metavar='N',
                        help='steps of pre-training of generators (default: 120)')
    parser.add_argument('--d_pretrain_steps', type=int, default=50, metavar='N',
                        help='steps of pre-training of discriminators (default: 50)')
    parser.add_argument('--g_steps', type=int, default=1, metavar='N',
                        help='steps of generator updates in one round of adverarial training (default: 1)')
    parser.add_argument('--d_steps', type=int, default=3, metavar='N',
                        help='steps of discriminator updates in one round of adverarial training (default: 3)')
    parser.add_argument('--gk_epochs', type=int, default=1, metavar='N',
                        help='epochs of generator updates in one step of generate update (default: 1)')
    parser.add_argument('--dk_epochs', type=int, default=3, metavar='N',
                        help='epochs of discriminator updates in one step of discriminator update (default: 3)')
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--update_rate', type=float, default=0.8, metavar='UR',
                        help='update rate of roll-out model (default: 0.8)')
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument('--n_rollout', type=int, default=16, metavar='N',
                        help='number of roll-out (default: 16)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='vocabulary size (default: 10)')
    parser.add_argument('--generator_name_or_path', type=str, default="Salesforce/codet5-base", metavar='PATH',
                        help='generator name or path (default: Salesforce/codet5-base)')
    parser.add_argument('--load_generator_path', type=str, default=None, metavar='PATH',
                        help='trained generator or path (default: None')
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_dis_seq_length", default=128, type=int,
                        help="The maximum total input sequence length of disciminator after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument('--discriminator_name_or_path', type=str, default="microsoft/codebert-base", metavar='PATH',
                        help='discriminator name or path (default: microsoft/codebert-base)')
    parser.add_argument('--load_discriminator_path', type=str, default=None, metavar='PATH',
                        help='trained discriminator or path (default: None')
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--n_samples', type=int, default=-1, metavar='N',
                        help='number of samples gerenated per time (default: -1)')
    parser.add_argument("--beam_size", default=3, type=int,
                        help="beam size for beam search")
    parser.add_argument("--evaluate_sample_size", default=-1, type=int,
                        help="sample size of evaluate.")
    parser.add_argument('--gen_lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate of generator optimizer (default: 1e-3)')
    parser.add_argument('--dis_lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate of discriminator optimizer (default: 1e-3)')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--gpu_id", type=int, default=-1,
                        help="only use single specified gpu for training")
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.gpu_id != -1 and torch.cuda.is_available():
        # torch.cuda.set_device(args.gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device("cuda:" + str(args.gpu_id))
        args.n_gpu = 1
    elif args.gpu_id != -1 and not torch.cuda.is_available():
        device = torch.device("cpu")
        args.n_gpu = 0
    elif args.no_cuda is True:
        device = torch.device("cpu")
        args.n_gpu = 0
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_count = multiprocessing.cpu_count()
    logger.warning("Device: %s, n_gpu: %s, cpu count: %d",
                   device, args.n_gpu, cpu_count)
    args.device = device
    args.cpu_count = cpu_count