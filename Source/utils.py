import json
import logging
import os
import random
import re
import subprocess as sp
import time
import numpy as np

import torch
from torch.utils.data import (ConcatDataset, DataLoader, Dataset,
                              RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from transformers import RobertaTokenizer, T5Tokenizer
from copy import deepcopy

logger = logging.getLogger(__name__)

def get_loader(data_file, args, tokenizer, pool, eval=False, batch_size=None, rand=False, save_purified_file=False, samplenum=-1):
    def fn(features):
        return features
    dataset = TextDataset(tokenizer, pool, args, data_file, save_purified_file=save_purified_file)
    data_len = len(dataset)
    if eval and samplenum == -1:
        sampler = SequentialSampler(dataset)
    elif samplenum != -1 and len(dataset) > samplenum:
        sampler = SubsetRandomSampler(random.sample(range(len(dataset)), samplenum))
    elif args.n_gpu > 1:
        sampler = DistributedSampler(dataset)
    elif eval is False and rand is True:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    logger.info(f"Sample size: {len(sampler)}.")
    if batch_size is None:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, num_workers=args.cpu_count, collate_fn=fn)
    else:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=args.cpu_count, collate_fn=fn)
    return dataset, sampler, dataloader


def generate_samples(args, pool, model, tokenizer, input_file, output_file, eval_flag=False, rand=False, sample_num=-1):
    res = []
    batch_size = args.eval_batch_size if args.train_batch_size else args.train_batch_size
    _, _, dataloader = get_loader(input_file, args, tokenizer, pool, eval=eval_flag, batch_size=batch_size,rand=rand, save_purified_file=True, samplenum=sample_num)
    bar = tqdm(dataloader, total=len(dataloader), desc="Generating samples")
    pred_ids, ori_doc_ids, indexes = [], [], []
    for step, examples in enumerate(bar, 1):
        if args.debug and step > 100:
            logger.info("Stop generation after 100 steps in DEBUG mode.")
            break
        doc_ids = torch.tensor([ex.doc for ex in examples], dtype=torch.long).to(args.device)
        code_ids = torch.tensor([ex.code for ex in examples], dtype=torch.long).to(args.device)
        ids = [ex.idx for ex in examples]
        gnerated_code_ids = model.my_sample(batch_size, args.max_target_length, x=doc_ids, y=None) # for generate from zero. the y is None
        pred_ids.extend(gnerated_code_ids)
        ori_doc_ids.extend(doc_ids)
        indexes.extend(ids)
    # [1:] to remove beginning '<msg>'
    pred_code_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    doc_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in ori_doc_ids]
    assert pred_code_nls.__len__() == doc_nls.__len__()
    for i in range(pred_code_nls.__len__()):
        res.append(json.dumps({
            "docstring":doc_nls[i],
            "code": pred_code_nls[i],
        }) + '\n')
    with open(output_file, 'w') as fout:
        for nl in res:
            fout.write(nl)
            
            
def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            try:
                js = json.loads(line.strip())
            except:
                print("Error during reading json data.")
                continue
            data.append(js)
    return data


class TextDataset(Dataset):

    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1, random_sample_num=-1, save_purified_file=False):
        self.cnt = 0
        self.tokenizer = tokenizer
        self.args = args
        if isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = ".rb_" + str(len(tokenizer))
        else:
            tokenizer_type = "unk"

        sample_tag = "all" if samplenum == -1 else str(samplenum) 
        savep = file_path.replace(".jsonl", tokenizer_type + "_" + sample_tag + ".exps")
        purified_jsonl_file = file_path.replace(".jsonl", "_purified.jsonl")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            self.feats = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            start = time.time()
            examples = read_examples(
                args, file_path, samplenum)
            end = time.time()
            logger.info(f"Read examples time cost: {end-start}")
            logger.info(f"Tokenize examples: {file_path}")
            if args.debug:
                self.convert_examples_to_features((examples[0], tokenizer, args))
            self.feats = pool.map(self.convert_examples_to_features,
                                [(example, tokenizer, args) for example in examples])
            torch.save(self.feats, savep)
        if save_purified_file is True:
            examples = read_examples(args, file_path, samplenum)
            res = []
            for ex in examples:
                res.append(json.dumps({
                    "docstring":ex.doc,
                    "code": ex.code,
                }) + '\n')
            with open(purified_jsonl_file, "w", encoding="utf8") as f:
                f.writelines(res)
            
        if args.debug:
            logger.info("Debug mode")
            logger.info(f"test random: {random.random()}")
            logger.info(f"Examples size: {self.feats.__len__()}")


    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

    def reset_len(self, data_len):
        assert len(self.feats) >= data_len
        self.feats = self.feats[:data_len]

    def convert_examples_to_features(self, item):
        example, tokenizer, args = item
        doc_ids = self.encode_remove(tokenizer, example.doc, limit_length=args.max_source_length)
        code_ids = self.encode_remove(tokenizer, example.code, limit_length=args.max_target_length)
        doc_ids, code_ids = self.pad_assert(doc_ids, code_ids, args, tokenizer)
        return SimpleExample(example.idx, doc_ids, code_ids)


    def pad_assert(self, source_ids, target_ids, args, tokenizer):
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_token_id] + source_ids + [tokenizer.eos_token_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * pad_len
        target_ids = target_ids[:args.max_target_length - 1]
        target_ids = target_ids + [tokenizer.bos_token_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
        return source_ids, target_ids


    def encode_remove(self, tokenizer, text, limit_length=None):
        if limit_length is None:
            text = tokenizer.encode(
                text)
        else:        
            text = tokenizer.encode(
                text, max_length=limit_length, truncation=True)
        if type(tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(tokenizer) == RobertaTokenizer:
            return text[1:-1]
        else:
            raise NotImplementedError


def read_examples(args, filename, data_num=-1):
    """Read examples from filename."""
    examples = []
    idx = 0
    items = read_jsonl(filename)
    
    for js in items:
        doc_tokens = js['docstring_tokens'] if "docstring_tokens" in js else ""
        code = js["code"] if "code" in js else ""
        try:
            doc = " ".join(doc_tokens)
        except:
            doc = ""
        example = SimpleExample(
            idx=idx,
            doc=doc,
            code=code
        )
        examples.append(example)
        idx += 1
        if idx == data_num:
            break
    return examples


def query_gpu_utilization(gpu_id=0):
    command = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
    res = sp.check_output(command.split()).decode("utf8").split('\n')
    if res.__len__() > (gpu_id + 1):
        return int(res[gpu_id+1].strip(" %"))
    return None


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def save_model(model, output_dir, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    config.save_pretrained(output_dir)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    
    
class SimpleExample(object):
    """A single example."""
    def __init__(self, idx, doc, code, doc_nls=None, code_nls=None, label=1):
        self.idx = idx
        self.doc = doc
        self.code = code
        self.doc_nls = doc_nls
        self.code_nls = code_nls
        self.label=label
        
        
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


def my_decode(tokenizer, ids, special_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().numpy().tolist()
    else:
        pass
    if skip_special_tokens is True:
        nl_ids = [x for x in ids if x not in special_ids ]
    else:
        nl_ids = ids
    decoded_string = tokenizer.decode(nl_ids, clean_up_tokenization_spaces=False)
    return decoded_string


def gen_masked_ids(tokenizer, ids, mask_rate, pad=False):
    source_ids, target_ids = [], []
    msg_ids = deepcopy(ids)
    masks = [random.random() < mask_rate for _ in range(len(msg_ids))]
    if sum(masks) == 0:
        idx = random.choice(range(len(msg_ids)))
        masks[idx] = True
    source_ids, target_ids = [], []
    i = 0
    SPECIAL_ID = 0
    while i < len(masks):
        j = i
        while j < len(masks) and not masks[j]:
            source_ids.append(msg_ids[j])
            j += 1
        if j == len(masks):
            break
        source_ids.append(tokenizer.special_dict[f"<extra_id_{SPECIAL_ID}>"])
        target_ids.append(tokenizer.special_dict[f"<extra_id_{SPECIAL_ID}>"])
        while j < len(masks) and masks[j]:
            target_ids.append(msg_ids[j])
            j += 1
        if SPECIAL_ID < 99:     # only 0-99 ids in vocab
            SPECIAL_ID += 1
        i = j
    if pad is True:
        src_pad_len = len(ids) - len(source_ids)
        tgt_pad_len = len(ids) - len(target_ids)
        source_ids += [tokenizer.pad_token_id]*src_pad_len
        target_ids += [tokenizer.pad_token_id]*tgt_pad_len
    return source_ids, target_ids
    
if __name__ == "__main__":
    print(query_gpu_utilization())
