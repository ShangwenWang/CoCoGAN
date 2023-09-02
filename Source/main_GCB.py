import argparse
from cProfile import label
from copy import deepcopy
import json
import multiprocessing
import os
import pickle as pkl
import time
import io
import tokenize
from numpy import source
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, T5Tokenizer
from configs import add_args, set_dist, set_seeds
from data_iter import DisDataIter, DisDataIterExpanded, GenDataIter
from discriminator import Discriminator
from torch.nn import CrossEntropyLoss, MSELoss
from evaluator.calc_crystalbleu import cal_crystalbleu
from evaluator.CodeBLEU import get_codebleu
from evaluator.smooth_bleu import bleu_fromstr
from generator import Generator
from loss import PGLoss
from models import build_or_load_gen_model
from rollout import Rollout
from GCB_utils import get_loader
from utils import save_model, my_decode, gen_masked_ids


def train_generator_MLE(gen, data_iter, criterion, optimizer, epochs, 
        gen_pretrain_train_loss, args):
    """
    Train generator with MLE
    """
    for epoch in range(epochs):
        total_loss = 0.
        for data, target in data_iter:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            output = gen(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        data_iter.reset()
    avg_loss = total_loss / len(data_iter)
    print("Epoch {}, train loss: {:.5f}".format(epoch, avg_loss))
    gen_pretrain_train_loss.append(avg_loss)


def train_generator_PG(train_dataloader, gen:Generator, dis:Discriminator, rollout:Rollout, pg_loss:PGLoss, nllloss, gen_optimizer, dis_optimizer, args):
    """
    Train generator with the guidance of policy gradient
    """
    global_step = 0
    nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
    gen.train()
    dis.train()
    bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Generator training")
    for step, examples in enumerate(bar, 1):
        # train discriminator
        doc_ids = examples[3].to(args.device)
        code_ids = examples[0][:,:args.max_target_length].to(args.device)
        code_ids_full = examples[0].to(args.device)
        attn_mask = examples[1].to(args.device)
        position_idx = examples[2].to(args.device)
        code_vec = dis(code_inputs=code_ids_full,attn_mask=attn_mask,position_idx=position_idx, convert=True)
        nl_vec = dis(nl_inputs=doc_ids, convert=True)
        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        dis_loss = loss_fct(scores, torch.arange(code_ids_full.size(0), device=scores.device))
        dis_loss.backward()
        torch.nn.utils.clip_grad_norm_(dis.parameters(), 1.0)
        dis_optimizer.step()
        dis_optimizer.zero_grad()
    
        rewards, output, cur_generated_examples = rollout.get_reward_simplified_v3_gcb(examples, args.n_rollout, dis) # shape: batchsize * sequence length
        loss = pg_loss(output, code_ids.view(-1), rewards)
        tr_loss += loss.item()
        nb_tr_steps += 1
        loss.backward()
        if nb_tr_steps % args.gradient_accumulation_steps == 0:
            # Update parameters
            gen_optimizer.step()
            gen_optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps , 4)
            bar.set_description("Train loss {}".format(round(train_loss, 3)))

    
def eval_generator_loss(generator, dataloader, args, output_path=None):
    
    """
    Evaluate generator with NLL
    """
    criterion = nn.NLLLoss(ignore_index=args.tokenizer_g.pad_token_id)
    total_loss = 0.
    bar = tqdm(dataloader, total=len(dataloader), desc="Generator eval")
    generator.eval()
    with torch.no_grad():
        for idx, examples in enumerate(bar, 1):
            doc_ids = examples[3].to(args.device)
            code_ids = examples[0][:,:args.max_target_length].to(args.device)
            log_probs = generator(doc_ids=doc_ids, code_ids=code_ids,
                           output_scores=True,
                )
            target = code_ids.contiguous().view(-1)
            loss = criterion(log_probs, target) # same as cross entropy loss
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    generator.train()
    
    
    return avg_loss



def eval_generator_bleu(generator, dataloader, args, output_path=None):
    """
    Evaluate generator with NLL
    """
    total_loss = 0.
    res = []
    pred_ids, ori_doc_ids, ori_code_ids, indexes = [], [], [], []
    bar = tqdm(dataloader, total=len(dataloader), desc="Generator eval")
    generator.eval()
    # ori_code_nls = []
    generated_examples = []
    special_ids = set(generator.tokenizer.all_special_ids)
    with torch.no_grad():
        for idx, examples in enumerate(bar, 1):
            doc_ids = examples[3].to(args.device)
            code_ids = examples[0][:,:args.max_target_length].to(args.device)
            output = generator.generate(doc_ids, 
                           do_sample=False,
                           num_beams=args.beam_size,
                           max_length = args.max_target_length,
                           output_scores=True,
                           return_dict_in_generate=True
            )
            samples = output.sequences
            pred_ids.extend(samples)
            ori_doc_ids.extend(doc_ids)
            ori_code_ids.extend(code_ids)

    generator.train()
    special_ids = set(generator.tokenizer.all_special_ids)
    pred_code_nls = [my_decode(generator.tokenizer, id, special_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    ori_code_nls = [my_decode(generator.tokenizer, id, special_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in ori_code_ids]
    ori_doc_nls = [my_decode(generator.tokenizer, id, special_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in ori_doc_ids]
    bleu = bleu_fromstr(ori_code_nls, pred_code_nls, rmstop=False)
    CodeBLEU = get_codebleu(ori_code_nls, pred_code_nls, args.language)
    crystalbleu = cal_crystalbleu(ori_code_nls, pred_code_nls, args.dev_filename, args.language)
    if output_path is not None:
        if output_path.endswith('.jsonl'):
            for i in range(pred_code_nls.__len__()):
                res.append(json.dumps({
                    "docstring":ori_doc_nls[i],
                    "code": pred_code_nls[i],
                }) + '\n')
                
            with open(output_path, "w", encoding="utf8") as fout:
                for nl in res:
                    fout.write(nl)
        elif output_path.endswith('.exps'):
            torch.save(generated_examples, output_path)
        else:
            pass
    return bleu, CodeBLEU, crystalbleu


def train_discriminator(dis:Discriminator, gen, criterion, optimizer, epochs, 
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Train discriminator
    """
    if isinstance(gen.tokenizer, T5Tokenizer):
        tokenizer_type = ""
    elif isinstance(gen.tokenizer, RobertaTokenizer):
        tokenizer_type = ".rb_" + str(len(gen.tokenizer))
    else:
        tokenizer_type = "unk"

    sample_tag = "all" if args.n_samples == -1 else str(args.n_samples) 
    positive_file = args.train_filename.replace(".jsonl", tokenizer_type + "_" + sample_tag + ".exps")
    negative_file = args.train_filename.replace(".jsonl", tokenizer_type + "_generated.exps")
    data_iter = DisDataIterExpanded(args, positive_file, negative_file, args.train_batch_size)
    for epoch in range(epochs):
        correct = 0
        total_loss = 0.
        nb_tr_steps, tr_loss = 0, 0
        bar = tqdm(data_iter, total=len(data_iter), desc="Discriminator training")
        for step, examples in enumerate(bar, 1):
       
            labels = torch.tensor([ex.label for ex in examples], dtype=torch.long).to(args.device)
            doc_ids = examples[3].to(args.device)
            code_ids = examples[0].to(args.device)
            logits = dis(comment_ids=doc_ids, code_ids=code_ids, convert=True)
            pred = logits.data.max(1)[1]
            correct += pred.eq(labels.data).cpu().sum()
            log_probs = F.log_softmax(logits, dim=1) 
            loss = criterion(log_probs, labels)
            tr_loss += loss.item()
            total_loss += loss.item()
            nb_tr_steps += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = round(tr_loss * 1 / nb_tr_steps , 4)
            bar.set_description("Train loss {}".format(round(train_loss, 3)))
        data_iter.reset()
        avg_loss = total_loss / nb_tr_steps
        acc = correct.item() / (nb_tr_steps*len(examples))
        print("Epoch {}, train loss: {:.5f}, train acc: {:.3f}".format(epoch, avg_loss, acc))
        dis_adversarial_train_loss.append(avg_loss)
        dis_adversarial_train_acc.append(acc)



def adversarial_train(dataloader, generator, discriminator, rollout, pg_loss, nll_loss, gen_optimizer, dis_optimizer, 
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Adversarially train generator and discriminator
    """
    # train generator for g_steps
    if args.do_train_generator:
        print("#Train generator")
        generator.train()
        for i in range(args.gk_epochs):
            print("##G-Step {}".format(i))
            train_generator_PG(dataloader, generator, discriminator, rollout, pg_loss, nll_loss, gen_optimizer, dis_optimizer, args)
    
    
    if args.do_train_discriminator:
        # train discriminator for d_steps
        print("#Train discriminator")
        for i in range(args.dk_epochs):
            print("##D-Step {}".format(i))
            train_discriminator(discriminator, generator, nll_loss, dis_optimizer, args.dk_epochs, 
                dis_adversarial_train_loss, dis_adversarial_train_acc, args)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    set_dist(args)
    set_seeds(args.seed)
    if args.debug:
        pool = multiprocessing.Pool(4)
        args.n_rollout = 1
    else:
        pool = multiprocessing.Pool(args.cpu_count)
    
    torch.manual_seed(args.seed)
    if not args.hpc:
        args.data_path = ''

    # Set models, criteria, optimizers
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    generator, tokenizer_g = build_or_load_gen_model(args, 'generator', args.generator_name_or_path)
    discriminator, tokenizer_d = build_or_load_gen_model(args, 'discriminator_gcb', args.discriminator_name_or_path)
    discriminator.tokenizer_g = tokenizer_g # assign a generator tokenizer to discriminator
    tokenizer_g.special_dict = {
        f"<extra_id_{i}>": tokenizer_g.get_vocab()[f"<extra_id_{i}>"] for i in range(99, -1, -1)
    }
    args.tokenizer_g = tokenizer_g
    args.tokenizer_d = tokenizer_d
    
    discriminator.special_ids = set(discriminator.tokenizer.all_special_ids)
    generator.special_ids = set(generator.tokenizer.all_special_ids)
    discriminator.g_special_ids = set(generator.tokenizer.all_special_ids)
    generator.d_special_ids = set(discriminator.tokenizer.all_special_ids)
    args.g_special_ids = generator.special_ids
    args.d_special_ids = discriminator.special_ids
    _, _, train_dataloader = get_loader(args.train_filename, args, tokenizer_g, pool, rand=True, samplenum=args.n_samples) 

    
    nll_loss = nn.NLLLoss()
    pg_loss = PGLoss(ignore_index=args.tokenizer_g.pad_token_id)
    nll_loss.to(args.device)
    pg_loss.to(args.device)
    cudnn.benchmark = True
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in generator.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in generator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    gen_optimizer = optim.Adam(params=generator.parameters(), lr=args.gen_lr)
    num_train_optimization_steps = args.gk_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(gen_optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)
    dis_optimizer = optim.AdamW(discriminator.parameters(), lr=args.dis_lr, eps=1e-8)

    # Container of experiment data
    gen_pretrain_train_loss = []
    gen_pretrain_eval_loss = []
    dis_pretrain_train_loss = []
    dis_pretrain_train_acc = []
    dis_pretrain_eval_loss = []
    dis_pretrain_eval_acc = []
    gen_adversarial_eval_loss = []
    dis_adversarial_train_loss = []
    dis_adversarial_train_acc = []
    dis_adversarial_eval_loss = []
    dis_adversarial_eval_acc = []

    # Adversarial training
    print('#####################################################')
    print('Start adversarial training...')
    print('#####################################################\n')
    rollout = Rollout(args, generator, args.update_rate)
    
    if isinstance(generator.tokenizer, T5Tokenizer):
        tokenizer_type = ""
    elif isinstance(generator.tokenizer, RobertaTokenizer):
        tokenizer_type = ".rb_" + str(len(generator.tokenizer))
    else:
        tokenizer_type = "unk"
    sample_tag = "all" if args.n_samples == -1 else str(args.n_samples) 
    positive_file = args.dev_filename.replace(".jsonl", tokenizer_type + "_" + sample_tag + ".exps")
    negative_file = args.dev_filename.replace(".jsonl", tokenizer_type + "_generated.exps")
    

    if args.do_train:
        for i in range(args.rounds):
            print("Round {}".format(i))
            adversarial_train(train_dataloader, generator, discriminator, rollout, 
                pg_loss, nll_loss, gen_optimizer, dis_optimizer, 
                dis_adversarial_train_loss, dis_adversarial_train_acc, args)
            
            if args.do_eval:
                _, _, gen_eval_dataloader = get_loader(args.dev_filename, args, tokenizer_g, pool,
                                                       eval=True, rand=True, batch_size=args.eval_batch_size,
                                                       save_purified_file=True, samplenum=args.evaluate_sample_size) 
                gen_loss = eval_generator_loss(generator, gen_eval_dataloader, args, output_path=negative_file)
                gen_bleu, gen_code_bleu, crystalbleu = eval_generator_bleu(generator, gen_eval_dataloader, args, output_path=negative_file)
                gen_adversarial_eval_loss.append(gen_loss)
                torch.save([gen_eval_dataloader.dataset.examples[i] for i in gen_eval_dataloader.sampler], positive_file)
                dis_acc = -1
                print("gen eval loss: {:.5f}, eval bleu: {:.5f}, eval CodeBLEU: {:.5f}, eval crystalbleu: {:.5f}, dis eval loss: {:.5f}, dis eval acc: {:.3f}\n"
                    .format(gen_loss, gen_bleu, gen_code_bleu, crystalbleu, -1, -1))
                gen_output_path = os.path.join(args.output_dir, "Gen", f"epoch_{i}" + "_" + str(int(gen_code_bleu*1000)/10))
                dis_output_path = os.path.join(args.output_dir, "Dis", f"epoch_{i}" + "_" + str(int(dis_acc*1000)/10))
            else:
                gen_output_path = os.path.join(args.output_dir, "Gen", f"epoch_{i}")
                dis_output_path = os.path.join(args.output_dir, "Dis", f"epoch_{i}")
            save_model(generator, gen_output_path, generator.config)
            save_model(discriminator, dis_output_path, discriminator.config)

    if args.do_test:
        sample_tag = "all" if args.evaluate_sample_size == -1 else str(args.evaluate_sample_size) 
        positive_file = args.test_filename.replace(".jsonl", tokenizer_type + "_" + sample_tag + ".exps")
        negative_file = args.test_filename.replace(".jsonl", tokenizer_type + "_generated_gcb_python.jsonl")
        _, _, gen_eval_dataloader = get_loader(args.test_filename, args, tokenizer_g, pool, 
                                               eval=True,rand=False, batch_size=args.eval_batch_size,
                                               save_purified_file=True, samplenum=args.evaluate_sample_size) 
        gen_bleu, code_bleu, crystalbleu = eval_generator_bleu(generator, gen_eval_dataloader, args, output_path=negative_file)
        print(code_bleu, crystalbleu)
