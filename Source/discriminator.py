import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification

from utils import _truncate_seq_pair, my_decode


class Discriminator(RobertaForSequenceClassification):
    """
    A Transformer for code classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Highway architecture based on the pooled feature maps is added. Dropout is adopted.
    """

    def __init__(self, num_classes, vocab_size, embedding_dim, filter_sizes, num_filters, dropout_prob):
        self.tokenizer = None
        self.tokenizer_g = None
        self.args = None
        super(Discriminator).__init__()
        # self.embed = nn.Embedding(vocab_size, embedding_dim)
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(1, num_f, (f_size, embedding_dim)) for f_size, num_f in zip(filter_sizes, num_filters)
        # ])
        # self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        # self.dropout = nn.Dropout(p = dropout_prob)
        # self.fc = nn.Linear(sum(num_filters), num_classes)
    
    @staticmethod
    def from_pretrained(path, config):
        model = RobertaForSequenceClassification.from_pretrained(path, config=config)
        model.__class__ = Discriminator
        return model

    def pad_list(self, tokenizer, input_ids, max_length):
        pad_len = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        return input_ids
        
    # def forward(self, x):
    #     """
    #     Inputs: x
    #         - x: (batch_size, seq_len)
    #     Outputs: out
    #         - out: (batch_size, num_classes)
    #     """
    #     emb = self.embed(x).unsqueeze(1) # batch_size, 1 * seq_len * emb_dim
    #     convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] # [batch_size * num_filter * seq_len]
    #     pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
    #     out = torch.cat(pools, 1)  # batch_size * sum(num_filters)
    #     highway = self.highway(out)
    #     transform = F.sigmoid(highway)
    #     out = transform * F.relu(highway) + (1. - transform) * out # sets C = 1 - T
    #     out = F.log_softmax(self.fc(self.dropout(out)), dim=1) # batch * num_classes
    #     return out
    def convert(self, old_ids, padding=False, return_tensor=False, max_length=-1):
        """converted ids to new ids tokenized by tokenizer.
        Args:
            old_ids (_type_): ids that require converted
        """
        device = self.device
        # nls = [self.tokenizer_g.decode(id[1:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in old_ids]
        # nls = [self.tokenizer_g.decode(id, clean_up_tokenization_spaces=False) for id in old_ids]
        nls = [my_decode(self.tokenizer_g, id, self.g_special_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in old_ids]
        # 不需要这行，因为tokenizer生成的就已经带bos和eos了
        # new_ids = [[self.tokenizer.bos_token_id] + self.tokenizer.encode(nl, max_length=length - 2, truncation=True) + [self.tokenizer.eos_token_id] for nl in nls] 
        new_ids = [self.tokenizer.encode(nl)for nl in nls]
        if padding:
            new_ids = [self.pad_list(self.tokenizer, x, max_length=max_length) for x in new_ids]
        if return_tensor:
            new_ids = torch.tensor(new_ids).to(device)
        return new_ids
    
    def construct_dis_inputs(self, doc_ids, code_ids, return_tensor=False):
        all_tokens = []
        for doc, code in zip(doc_ids, code_ids):
            if doc.__len__() > 0 and doc[0] == self.tokenizer.bos_token_id and doc[-1] == self.tokenizer.eos_token_id:
                doc = doc[1:-1]
            if code.__len__() > 0 and code[0] == self.tokenizer.bos_token_id and code[-1] == self.tokenizer.eos_token_id:
                code = code[1:-1]
            doc = doc[:50]
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(doc, code, self.args.max_dis_seq_length-3)
            tokens = [self.tokenizer.cls_token_id] + doc + [self.tokenizer.sep_token_id] + code + [self.tokenizer.sep_token_id]
            all_tokens.append(tokens)
        if return_tensor:
            all_tokens = [self.pad_list(self.tokenizer, x, max_length=self.args.max_dis_seq_length) for x in all_tokens]
            return torch.tensor(all_tokens)
        else:
            return all_tokens
    
            
    # def forward(self, comment_ids, code_ids, convert=False):
    def forward(self, *argv, **kwargs):
        if "comment_ids" in kwargs:
            comment_ids = kwargs["comment_ids"]
            code_ids = kwargs["code_ids"]
            if "convert" in kwargs and kwargs["convert"] is True:
                # comment_ids = self.convert(comment_ids, padding=False, return_tensor=False, max_length=self.args.max_source_length)
                # code_ids = self.convert(code_ids, padding=False, return_tensor=False, max_length=self.args.max_target_length)
                # test
                comment_ids = self.convert(comment_ids, padding=False, return_tensor=False, max_length=self.args.max_source_length)
                code_ids = self.convert(code_ids, padding=False, return_tensor=False, max_length=self.args.max_target_length)
                ####################### 
            input_ids = self.construct_dis_inputs(comment_ids, code_ids, return_tensor=True).to(self.device)
            attention_mask = input_ids.ne(self.config.pad_token_id).to(self.device)
            labels = torch.zeros((input_ids.size(0)),dtype=torch.int64).to(self.device)
            inputs = {'input_ids': input_ids,
                          'attention_mask': attention_mask,
                          'token_type_ids': None,
                          # XLM don't use segment_ids
                          'labels': labels}
            outputs = self(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            # out = F.log_softmax(logits, dim=1) # batch * num_classes, 如果是要cross entropy的话应该是要用log softmax的
            # out = F.softmax(logits, dim=1)  # TODO: 试试用哪一个
            # out = torch.sigmoid(logits)  # TODO: 试试用哪一个
            return logits
        return super().forward(*argv, **kwargs)
