import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from discriminator import Discriminator
from utils import query_gpu_utilization
from GCB_utils import extract_features_from_code_ids

class Rollout(object):
    """ Rollout Policy """

    def __init__(self, args, model, update_rate):
        self.args = args
        self.own_model = model
        self.update_rate = update_rate


    def get_reward_simplified_v3(self, examples, num, discriminator: Discriminator):
        """
        Inputs: x, num, discriminator
            - x: (batch_size, seq_len) input data
            - num: rollout number
            - discriminator: discrimanator model
        """
        doc_ids = torch.tensor(
                [ex.doc for ex in examples], dtype=torch.long
            ).to(self.own_model.device)
        code_ids = torch.tensor(
                [ex.code for ex in examples], dtype=torch.long
            ).to(self.own_model.device)
        batch_size = doc_ids.shape[0]
        gnerated_code_ids = []
        output = self.own_model(doc_ids=doc_ids, code_ids=code_ids, output_scores=True)
        prob = torch.exp(output)
        gnerated_code_ids = torch.multinomial(prob, 1).view((batch_size, self.args.max_target_length))
        with torch.no_grad():
            logits = discriminator(
                comment_ids=doc_ids,
                code_ids=gnerated_code_ids,
                convert=True) 
        pred = torch.sigmoid(logits)
        pred = pred.cpu().data[:, 0].numpy()
        rewards = \
            torch.tensor(pred).to(self.args.device).view((batch_size, 1)) \
            * torch.ones((batch_size, self.args.max_target_length)).to(self.args.device)
        gnerated_code_ids = gnerated_code_ids.cpu().numpy().tolist()
        generated_examples = []
        for i, ex in enumerate(copy.deepcopy(examples)):
            ex.code = gnerated_code_ids[i]
            ex.label = 0 # negative examples
            generated_examples.append(ex)
        return rewards, output, generated_examples


    def get_reward_simplified_v3_gcb(self, examples, num, discriminator: Discriminator):
        """
        Inputs: x, num, discriminator
            - x: (batch_size, seq_len) input data
            - num: rollout number
            - discriminator: discrimanator model
        """
        doc_ids = examples[3].to(self.args.device)
        code_ids = examples[0][:,:self.args.max_target_length].to(self.args.device)
        code_ids_full = examples[0].to(self.args.device)
        batch_size = doc_ids.shape[0]
        gnerated_code_ids = []
        output = self.own_model(doc_ids=doc_ids, code_ids=code_ids, output_scores=True)
        prob = torch.exp(output)
        gnerated_code_ids = torch.multinomial(prob, 1).view((batch_size, self.args.max_target_length))
        
        # generate features
        code_inputs, attn_mask, position_idx = extract_features_from_code_ids(self.args, gnerated_code_ids, self.args.tokenizer_g)
        
        with torch.no_grad():
            code_vec = discriminator(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx, convert=True)
            nl_vec = discriminator(nl_inputs=doc_ids, convert=True)
        scores=torch.einsum("ab,cb->ac", nl_vec, code_vec) # size: (nl_vec.shape[0], code_vec.shape[0])
        pred = 1-np.array([scores[i][i].item() for i in range(code_ids_full.size(0))])/100 

        rewards = \
            torch.tensor(pred).to(self.args.device).view((batch_size, 1)) \
            * torch.ones((batch_size, self.args.max_target_length)).to(self.args.device)
        gnerated_code_ids = gnerated_code_ids.cpu().numpy().tolist()
        generated_examples = []
        for i, ex in enumerate(copy.deepcopy(examples)):
            ex.code = gnerated_code_ids[i]
            ex.label = 0 # negative example
            generated_examples.append(ex)
            
        return rewards, output, generated_examples
