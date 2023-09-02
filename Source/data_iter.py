import math
import multiprocessing
import random

import torch

from utils import TextDataset, SimpleExample


class GenDataIter:
    """ Toy data iter to load digits """

    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = math.ceil(self.data_num / self.batch_size)
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.tensor(d)

        # 0 is prepended to d as start symbol
        data = torch.cat([torch.zeros(len(index), 1, dtype=torch.int64), d], dim=1)
        target = torch.cat([d, torch.zeros(len(index), 1, dtype=torch.int64)], dim=1)
        
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = [int(s) for s in list(line.strip().split())]
            lis.append(l)
        return lis


class DisDataIter:
    """ Toy data iter to load digits """

    def __init__(self, args, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.args = args
        self.batch_size = batch_size
        self.data = []
        if args.debug:
            pool = multiprocessing.Pool(4)
        else:
            pool = multiprocessing.Pool(args.cpu_count)
        if real_data_file.endswith(".jsonl"):
            real_data_lis = TextDataset(args.tokenizer_d, pool, args, real_data_file)
        else:
            real_data_lis = torch.load(real_data_file)
        if fake_data_file.endswith(".jsonl"):
            fake_data_lis = TextDataset(args.tokenizer_d, pool, args, fake_data_file)
        else:
            fake_data_lis = torch.load(fake_data_file)
        # self.labels = [1 for _ in range(len(real_data_lis))] +\
        #                 [0 for _ in range(len(fake_data_lis))]
        for ex in real_data_lis:
            ex.label = 1
        for ex in fake_data_lis:
            ex.label = 0
        if hasattr(real_data_lis, "feats"):
            self.data += real_data_lis.feats
        else:
            self.data += real_data_lis
        if hasattr(fake_data_lis, "feats"):
            self.data += fake_data_lis.feats
        else:
            self.data += fake_data_lis
        # self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.data)
        self.indices = range(self.data_num)
        self.num_batches = math.ceil(self.data_num / self.batch_size)
        self.idx = 0
        self.reset()

    def add_negative_items(self,):
        pass
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.data)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        data = [self.data[i] for i in index]
        self.idx += self.batch_size
        return data

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = [int(s) for s in list(line.strip().split())]
            lis.append(l) 
        return lis


class DisDataIterExpanded:
    """ Toy data iter to load digits """

    def __init__(self, args, real_data_file, fake_data_file, batch_size, expanded = 5):
        super(DisDataIterExpanded, self).__init__()
        self.args = args
        self.batch_size = batch_size
        self.data = []
        
        if args.similarity_filename is not None:
            self.similarity = torch.load(args.similarity_filename)
        else:
            self.similarity = None
        if args.debug:
            pool = multiprocessing.Pool(4)
        else:
            pool = multiprocessing.Pool(args.cpu_count)
        if real_data_file.endswith(".jsonl"):
            real_data_lis = TextDataset(args.tokenizer_d, pool, args, real_data_file)
        else:
            real_data_lis = torch.load(real_data_file)
        # self.labels = [1 for _ in range(len(real_data_lis))] +\
        #                 [0 for _ in range(len(fake_data_lis))]
        for ex in real_data_lis:
            ex.label = 1
        if hasattr(real_data_lis, "feats"):
            self.data += real_data_lis.feats
        else:
            self.data += real_data_lis
        # self.pairs = list(zip(self.data, self.labels))
        expanded_data = []
        for i, data in enumerate(self.data):
            doc_ids = data.doc
            for _ in range(expanded):
                if self.similarity:
                    rand_idx = random.choice(self.similarity[i])
                else:
                    rand_idx = random.randint(0, len(self.data) -1)
                rand_data = self.data[rand_idx]
                code_ids = rand_data.code
                tmp_example = SimpleExample(i + 1000000 + _, doc_ids, code_ids)
                if rand_idx != i:
                    tmp_example.label = 0
                else:
                    tmp_example.label = 1
                expanded_data.append(tmp_example)
        self.data += expanded_data
        self.data_num = len(self.data)
        self.indices = range(self.data_num)
        self.num_batches = math.ceil(self.data_num / self.batch_size)
        self.idx = 0
        self.reset()

    def add_negative_items(self,):
        pass
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.data)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        data = [self.data[i] for i in index]
        self.idx += self.batch_size
        return data

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = [int(s) for s in list(line.strip().split())]
            lis.append(l) 
        return lis
