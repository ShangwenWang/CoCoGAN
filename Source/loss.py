import torch
import torch.nn as nn


class PGLoss(nn.Module):
    """
    Pseudo-loss that gives corresponding policy gradients (on calling .backward()) 
    for adversial training of Generator
    """

    def __init__(self, ignore_index=-100):
        self.ignore_index=ignore_index
        super(PGLoss, self).__init__()

    def forward(self, pred, target, reward):
        """
        Inputs: pred, target, reward
            - pred: (batch_size, seq_len), 
            - target : (batch_size, seq_len), 
            - reward : (batch_size, ), reward of each whole sentence
        """
        one_hot = torch.zeros(pred.size(), dtype=torch.bool)  # 这儿的dtype=uint8要修改可能？
        if pred.is_cuda:
            one_hot = one_hot.to(pred.device)
        one_hot.scatter_(1, target.data.view(-1, 1), True)
        reward = reward.contiguous().view(-1)
        if self.ignore_index != -100:
            one_hot.scatter_(1, torch.ones_like(target.data.view(-1, 1))*self.ignore_index, False) # ignore the ignore_index
            valid_indexes = torch.nonzero(target!=self.ignore_index).view(-1)
            reward = reward.index_select(0, valid_indexes)
        loss = torch.masked_select(pred, one_hot)
        loss = loss * reward
        loss = -torch.sum(loss)
        # loss = torch.sum(loss)
        return loss
