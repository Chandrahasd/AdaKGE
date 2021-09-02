import os
import numpy as np
import torch

from torch.utils.data import Dataset

from utils.load_data import LoadData

class DataPipeline(Dataset):
    '''
        Data Pipeline class which defines train and test batch data input function.
    '''
    def __init__(self, params, type='train', data_loader=None, mode='tail_prediction'):
        '''
            1. Triplets as a np array where each item is [h,r,t] with their ids,
            2. rh2t[relation_id][head_entity_id] = [list of tail ids] and
            3. Dictionary of params containing ent_vocab_size, batch_size
        '''
        data                       = data_loader
        data_params                = data.load_vocab_size()
        params.ent_vocab_size      = data_params['ent_vocab_size']
        params.rel_vocab_size      = data_params['rel_vocab_size']
        params.rel_vocab_size_half = data_params['rel_vocab_size_half']
        self.params                = params
        self.type                  = type
        self.entities              = np.arange(self.params.ent_vocab_size,dtype=np.int32)

        self.int_dtype             = np.int64
        self.float_dtype           = np.float32

        if type == 'train':
                self.triples         = data.load_train_triples(self.params.file_numbers, mode)
        elif type == 'valid':
                self.triples         = data.load_test_triples('valid',mode)
        elif type == 'test':
                self.triples         = data.load_test_triples('test', mode)

        if type == 'train':
            params.steps_per_epoch       = len(self.triples)//params.batch_size
            self.params.steps_per_epoch  = params.steps_per_epoch

    def __len__(self):
        return len(self.triples)

    def get_label(self, triple, label, is_tail_prediction):
        y            = np.zeros([self.params.ent_vocab_size], dtype =self.float_dtype)
        label        = np.asarray(label, dtype=np.int32)
        y[label] = 1.0
        return y

    def get_negative_entities(self, triple, label, is_tail_prediction):
        positive_entity = triple[2] if is_tail_prediction else triple[0]
        mask = np.ones([self.params.ent_vocab_size], dtype=np.bool)
        label = np.asarray(label, dtype=np.int32)
        mask[label] = 0
        negative_entities = np.asarray(np.random.choice(self.entities[mask], self.params.ent_x_size, replace=False)).reshape([-1])
        negative_entities = np.concatenate((positive_entity.reshape([-1]), negative_entities))
        return np.asarray(negative_entities, dtype=self.int_dtype)

    def get_label_and_negative_entities(self, triple, label, is_tail_prediction):
        mask = np.ones([self.params.ent_vocab_size], dtype=np.bool)
        label = np.asarray(label, dtype=np.int32)
        mask[label] = 0
        negative_entities = np.asarray(np.random.choice(self.entities[mask], self.params.ent_x_size, replace=False)).reshape([-1])
        negative_entities = np.concatenate((label.reshape([-1]), negative_entities))
        y = np.zeros([len(negative_entities)], dtype=self.float_dtype)
        y[:len(label)] = 1.
        return y, np.asarray(negative_entities, dtype=self.int_dtype)

    @staticmethod
    def collate_fn(data):
        '''data is List of samples'''
        h                  = torch.cat([_[0] for _ in data], dim   =0)
        r                  = torch.cat([_[1] for _ in data], dim   =0)
        t                  = torch.cat([_[2] for _ in data], dim   =0)
        negative_entities  = torch.stack([_[3] for _ in data], dim =0)
        subsampling_weight = torch.cat([_[4] for _ in data], dim   =0)
        mode               = data[0][5]
        output  = torch.stack([_[6] for _ in data], dim =0)
        return (h, r, t, negative_entities, subsampling_weight, mode), output

    @staticmethod
    def collate_fn_nx(data):
        '''data is List of samples'''
        h                  = torch.cat([_[0] for _ in data], dim   =0)
        r                  = torch.cat([_[1] for _ in data], dim   =0)
        t                  = torch.cat([_[2] for _ in data], dim   =0)
        max_len_ne         = max([len(_[3]) for _ in data])+1
        max_len_y          = max([len(_[6]) for _ in data])+1
        negative_entities  = torch.stack([torch.cat((_[3],_[3][-1].repeat(max_len_ne-len(_[3])))) for _ in data], dim =0)
        subsampling_weight = torch.cat([_[4] for _ in data], dim   =0)
        mode               = data[0][5]
        output             = torch.stack([torch.cat((_[6],_[6][-1].repeat(max_len_ne-len(_[6])))) for _ in data], dim =0)
        return (h, r, t, negative_entities, subsampling_weight, mode), output



'''DataPipeline 1TON'''
class DataPipeline_Train_Tail_1toN(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='tail_prediction')

    def __getitem__(self, index):
        sample                            = self.triples[index]
        triple, label, subsampling_weight = self.int_dtype(sample['triple']), self.float_dtype(sample['label']), self.float_dtype(sample['subsampling_weight'])
        h, r, t                           = triple
        h, r, t                           = np.asarray(h).reshape([-1]), np.asarray(r).reshape([-1]), np.asarray(t).reshape([-1])
        y                                 = self.get_label(triple, label, True)
        subsampling_weight                = np.asarray(subsampling_weight, dtype=self.float_dtype).reshape([-1])
        if self.params.is_label_smoothing == True:
                y                         = (1.0 - self.params.label_smoothing_epsilon)*y + (1.0/self.params.ent_vocab_size)
        return torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(np.zeros([1], dtype=self.int_dtype)), torch.from_numpy(subsampling_weight), 'tail_prediction', torch.from_numpy(y)

class DataPipeline_Test_Tail_1toN(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='tail_prediction')

    def __getitem__(self, index):
        sample                            = self.triples[index]
        triple, label                     = self.int_dtype(sample['triple']), self.float_dtype(sample['label'])
        h, r, t                           = triple
        h, r, t                           = np.asarray(h).reshape([-1]), np.asarray(r).reshape([-1]), np.asarray(t).reshape([-1])
        y                                 = self.get_label(triple, label, True)
        return torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(np.zeros([1], dtype=self.int_dtype)), torch.from_numpy(np.zeros([1], dtype=self.float_dtype)), 'tail_prediction', torch.from_numpy(y).byte()

class DataPipeline_Test_Head_1toN(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='head_prediction')

    def __getitem__(self, index):
        sample                            = self.triples[index]
        triple, label                     = self.int_dtype(sample['triple']), self.float_dtype(sample['label'])
        h, r, t                           = triple
        h, r, t                           = np.asarray(h).reshape([-1]), np.asarray(r).reshape([-1]), np.asarray(t).reshape([-1])
        y                                 = self.get_label(triple, label, False)
        prediction                        = 'head_prediction'
        h, t                              = t, h
        if self.params.reverse == True:
            r                  = r + self.params.rel_vocab_size_half
            prediction         = 'tail_prediction'
        return torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(np.zeros([1], dtype=self.int_dtype)), torch.from_numpy(np.zeros([1], dtype=self.float_dtype)), prediction, torch.from_numpy(y).byte()




'''DataPipeline 1TOX'''
class DataPipeline_Train_Tail_1toX(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='tail_prediction')

    def __getitem__(self, index):
        sample                            = self.triples[index]
        triple, label, subsampling_weight = self.int_dtype(sample['triple']), self.float_dtype(sample['label']), self.float_dtype(sample['subsampling_weight'])
        h, r, t                           = triple
        h, r, t                           = np.asarray(h).reshape([-1]), np.asarray(r).reshape([-1]), np.asarray(t).reshape([-1])
        negative_entities                 = self.get_negative_entities(triple, label, True)
        subsampling_weight                = np.asarray(subsampling_weight, dtype=self.float_dtype).reshape([-1])
        y                                 = [1]; y.extend([0]*self.params.ent_x_size)
        y                                 = np.asarray(y, dtype=self.float_dtype)
        if self.params.is_label_smoothing == True:
                y                         = (1.0 - self.params.label_smoothing_epsilon)*y + (1.0/self.params.ent_vocab_size)
        return torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(negative_entities), torch.from_numpy(subsampling_weight), 'tail_prediction', torch.from_numpy(y)

class DataPipeline_Train_Head_1toX(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='head_prediction')

    def __getitem__(self, index):
        sample                            = self.triples[index]
        triple, label, subsampling_weight = self.int_dtype(sample['triple']), self.float_dtype(sample['label']), self.float_dtype(sample['subsampling_weight'])
        h, r, t                           = triple
        h, r, t                           = np.asarray(h).reshape([-1]), np.asarray(r).reshape([-1]), np.asarray(t).reshape([-1])
        negative_entities                 = self.get_negative_entities(triple, label, False)
        subsampling_weight                = np.asarray(subsampling_weight, dtype=self.float_dtype).reshape([-1])
        y                                 = [1]; y.extend([0]*self.params.ent_x_size)
        y                                 = np.asarray(y, dtype=self.float_dtype)
        h,t                               = t,h
        if self.params.is_label_smoothing == True:
                y                         = (1.0 - self.params.label_smoothing_epsilon)*y + (1.0/self.params.ent_vocab_size)
        return torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(negative_entities), torch.from_numpy(subsampling_weight), 'head_prediction', torch.from_numpy(y)

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.dataloader_head = dataloader_head
        self.dataloader_tail = dataloader_tail
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    def __iter__(self):
        self.iterator_head = iter(self.dataloader_head)
        self.iterator_tail = iter(self.dataloader_tail)
        return self


'''DataPipeline 1TONX'''
class DataPipeline_Train_Tail_1toNX(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='tail_prediction')

    def __getitem__(self, index):
        sample                            = self.triples[index]
        triple, label, subsampling_weight = self.int_dtype(sample['triple']), self.float_dtype(sample['label']), self.float_dtype(sample['subsampling_weight'])
        h, r, t                           = triple
        h, r, t                           = np.asarray(h).reshape([-1]), np.asarray(r).reshape([-1]), np.asarray(t).reshape([-1])
        y, negative_entities              = self.get_label_and_negative_entities(triple, label, True)
        subsampling_weight                = np.asarray(subsampling_weight, dtype=self.float_dtype).reshape([-1])
        if self.params.is_label_smoothing == True:
                y                         = (1.0 - self.params.label_smoothing_epsilon)*y + (1.0/self.params.ent_vocab_size)
        return torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(negative_entities), torch.from_numpy(subsampling_weight), 'tail_prediction', torch.from_numpy(y)
