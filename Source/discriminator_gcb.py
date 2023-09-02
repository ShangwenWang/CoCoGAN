import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from utils import _truncate_seq_pair, my_decode


class Discriminator_gcb(nn.Module):   
    def __init__(self, encoder):
        super(Discriminator_gcb, self).__init__()
        self.encoder = encoder

    def convert(self, old_ids, padding=False, return_tensor=False, max_length=-1):
        """converted ids to new ids tokenized by tokenizer.
        Args:
            old_ids (_type_): ids that require converted
        """
        device = self.encoder.device
        nls = [my_decode(self.tokenizer_g, id, self.g_special_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in old_ids]
        new_ids = [self.tokenizer.encode(nl)for nl in nls]
        if padding:
            new_ids = [self.pad_list(self.tokenizer, x, max_length=max_length) for x in new_ids]
        if return_tensor:
            new_ids = torch.tensor(new_ids).to(device)
        return new_ids
    
    def pad_list(self, tokenizer, input_ids, max_length):
        input_ids = input_ids[:max_length]
        pad_len = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        return input_ids
    
    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None, convert=False): 
        if code_inputs is not None:
            if convert == True:
                code_inputs = self.convert(code_inputs, padding=True, return_tensor=True, max_length=code_inputs.size(1))
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else:
            if convert == True:
                nl_inputs = self.convert(nl_inputs, padding=True, return_tensor=True, max_length=nl_inputs.size(1))
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

