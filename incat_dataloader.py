import numpy as np
import copy 
import torch 

class InCategoryClutterDataloader(object):
    def __init__(self, dataset, batch_size, shuffle = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.num_batches, self.last_batch_size, self.batch_indices = self.assign_idx_to_batch()
        self.acc = 0
    
    def assign_idx_to_batch(self):
        batch_size = self.batch_size
        num_batches = int(len(self.dataset) / batch_size) + 1
        last_batch_size = len(self.dataset) - (num_batches - 1) * batch_size

        batch_indices = np.ones((num_batches, batch_size)).astype('int') * -1
        idx_dict = copy.deepcopy(self.dataset.object_id_to_dict_idx)
        
        fill_in_count = np.zeros(num_batches).astype(int)
        for k,v in idx_dict:
            if self.shuffle:
                np.random.shuffle(v)
            
            v_idx = 0
            while v_idx < len(v):
                for batch_idx in range(num_batches):
                    if v_idx >= len(v):
                        break
                    start = fill_in_count[batch_idx]
                    if start > batch_size-1 or (batch_idx == num_batches-1 and start > last_batch_size-1):
                        print(start)
                        continue 
                    batch_indices[batch_idx][start] = v[v_idx]
                    batch_indices[batch_idx][start+1] = v[v_idx+1]
                    fill_in_count[batch_idx] += 2
                    v_idx += 2
        if self.shuffle:
            for i in range(num_batches):
                if i == num_batches - 1:
                    np.random.shuffle(batch_indices[i])
                else:
                    np.random.shuffle(batch_indices[i])
        return num_batches, last_batch_size, batch_indices
    
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.acc = 0
        return self 
    
    def compile_batch(self, indices):
        all_data = {}

        for i in indices:
            if i < 0:
                break
            data = self.dataset[i]
            for j in range(len(data)):
                l = all_data.get(j,[])
                l.append(data[j])
                all_data[j] = l

        res = []
        for l in all_data.values(): 
            res.append(torch.stack(l, dim=0))

        return res

    
    def __next__(self):
        if self.acc < self.num_batches:
            data = self.compile_batch(self.batch_indices[self.acc])
            self.acc +=1
            return data
        else:
            raise StopIteration






# # available = np.full((num_batches, batch_size), True, dtype=bool)
# # available[-1, last_batch_size:] = False 
# batch_indices = np.ones((num_batches, batch_size)).astype('int') * -1
# all_keys = np.array(list(dataset.object_id_to_dict_idx.keys()))
# all_keys = np.random.permutation(all_keys)

# available_keys = np.full(len(dataset.object_id_to_dict_idx[k]), True, dtype=bool)
# available_dict = {}
# for k in all_keys:
#     available_dict[k] = np.full(len(dataset.object_id_to_dict_idx[k]), True, dtype=bool)

# for bi in range(num_batches):
#     if bi == num_batches-1:
#         times = last_batch_size // 2
#     else:
#         times = batch_size // 2
#     for j in range(times):
#         done = False 
#         while not done:
#             select_k = np.random.choice(all_keys, 1)[0]
#             available_arr = available_dict[select_k]
#             available_indices = np.where(available_arr)[0]
#             if len(available_indices) == 1:
#                 print(select_k, bi, j)
#                 assert False
#             if len(available_indices) == 0:
#                 continue
#             i1,i2 = np.random.choice(available_indices, 2, replace=False)
#             batch_indices[2*j] = i1 
#             batch_indices[2*j+1] = i2  
#             available_arr[i1] = False 
#             available_arr[i2] = False 
#             available_dict[select_k] = available_arr
#             done = True 

# # for k in all_keys:
# #     v = dataset.object_id_to_dict_idx[k]
# #     b = len(v) // 2
# #     for j in range(b):
# #         done = False
# #         while not done:
# #             batch_choice = np.random.choice(len(available), 1)[0]
# #             if np.sum(available[batch_choice]) <= batch_size-2:
# #                 continue
# #             batch_indices