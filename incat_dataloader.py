import numpy as np
import copy 
import torch 
import torch.distributed as dist

class InCategoryClutterDataloader(object):
    def __init__(self, dataset, args, shuffle = True):
        self.args = args
        self.dataset = dataset
        self.batch_size = args.training_config.batch_size
        self.shuffle = shuffle 
        
        self.acc = 0
        self.seed = args.training_config.seed
        self.epoch = 0

        if args.num_gpus > 1:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            if self.rank >= self.num_replicas or self.rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval"
                    " [0, {}]".format(self.rank, self.num_replicas - 1))
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = True

        self.num_batches, self.batch_indices = self.assign_idx_to_batch()
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def assign_idx_to_batch(self):
        np.random.seed(self.epoch + self.seed)

        if self.dataset.split == "train":
            self.dataset.reset(self.epoch + self.seed)
        batch_size = self.batch_size
        num_batches = int(len(self.dataset) / batch_size)

        batch_indices = np.ones((num_batches, batch_size)).astype('int') * -1
        idx_dict = copy.deepcopy(self.dataset.object_id_to_dict_idx)

        l = np.arange(num_batches * (batch_size // 2))
        if self.shuffle:
            np.random.shuffle(l)
        l = l.reshape(num_batches, batch_size//2)
        print(self.rank, l[0])
        
        vs = []
        for k,v in idx_dict.items():
            if self.shuffle:
                np.random.shuffle(v)
            vs.append(v)
        vs = np.hstack(vs).reshape(-1,2)
        if self.shuffle:
            np.random.shuffle(vs)
        else:
            width = 256
            A = np.array(list(np.arange(len(vs))) + [-1] * (width - len(vs) % width)).reshape(-1,width).T
            A = A.flatten()
            A = A[A > -1]
            vs = vs[A]
        
        for batch_idx in range(num_batches):
            for i in range(batch_size // 2):
                acc = l[batch_idx][i]
                v1,v2 = vs[acc]
                batch_indices[batch_idx][2*i] = v1
                batch_indices[batch_idx][2*i+1] = v2
        
        if self.shuffle:
            for i in range(num_batches):
                np.random.shuffle(batch_indices[i])
        
        if self.args.num_gpus > 1:
            per_replica = num_batches // self.num_replicas
            start_idx = per_replica * self.rank
            if self.rank == self.num_replicas-1:
                end_idx = num_batches
                num_batches = num_batches - per_replica * (self.num_replicas-1)
                
            else:
                num_batches = per_replica
                end_idx = per_replica * (self.rank+1)
            batch_indices = batch_indices[start_idx : end_idx]
                
        return num_batches, batch_indices
    
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.acc = 0
        self.num_batches, self.batch_indices = self.assign_idx_to_batch()
        return self 
    
    def compile_batch(self, indices):
        all_data = {}

        for i in indices:
            if i < 0:
                break
            data = self.dataset[i]
            for k,v in data.items():
                l = all_data.get(k,[])
                l.append(v)
                all_data[k] = l

        res = dict()
        for k,l in all_data.items():
            res[k] = torch.stack(l, dim=0)

        return res

    
    def __next__(self):
        if self.acc < self.num_batches:
            data = self.compile_batch(self.batch_indices[self.acc])
            self.acc +=1
            return data
        else:
            raise StopIteration