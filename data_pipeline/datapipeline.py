import time
import numpy as np
import torch

import math

from utils.load_data import LoadData
import multiprocessing as mp
from functools import partial

class DataPipeline(object):
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

    def get_label(self, triple, label, is_tail_prediction):
        y            = np.zeros([len(label), self.params.ent_vocab_size], dtype =self.float_dtype)
        for i, e2 in enumerate(label):
            y[i, e2] = 1.0
        return y

    def get_negative_entities(self, triple, label, is_tail_prediction):
        def get(triple, label):
            positive_entity   = triple[2] if is_tail_prediction else triple[0]
            mask              = np.ones([self.params.ent_vocab_size], dtype=np.bool)
            label             = np.asarray(label, dtype=np.int32)
            mask[label]       = 0
            negative_entities = np.asarray(np.random.choice(self.entities[mask], self.params.ent_x_size, replace=False)).reshape([-1])
            negative_entities = np.concatenate((positive_entity.reshape([-1]), negative_entities))
            return negative_entities
        negative_entities     = np.asarray(list(map(get,triple,label)), dtype=self.int_dtype)
        return negative_entities

    def get_negative_entities_mp(self, triple, label, is_tail_prediction):
        pool                = mp.Pool(self.params.num_data_workers)
        fn                  = partial(get_negative_entities_single, ent_vocab_size=self.params.ent_vocab_size, ent_x_size=self.params.ent_x_size, is_tail_prediction=is_tail_prediction, entities = self.entities)
        negative_entities   = np.asarray(list(pool.starmap(fn, zip(triple, label))), dtype=self.int_dtype)
        pool.close()
        return negative_entities

    def get_label_and_negs(self, triple, label, is_tail_prediction):
        label_lengths = [len(y) for y in label]
        max_len = max(label_lengths)+self.params.ent_x_size
        def get(triple, label):
            negative_length   = max_len - len(label)
            mask              = np.ones([self.params.ent_vocab_size], dtype=np.bool)
            label             = np.asarray(label, dtype=np.int32)
            mask[label]       = 0
            negative_entities = np.asarray(np.random.choice(self.entities[mask], self.params.ent_x_size, replace=False)).reshape([-1])
            if negative_length > self.params.ent_x_size:
                negative_entities_new = np.random.choice(negative_entities, negative_length-self.params.ent_x_size, replace=True).reshape([-1])
                negative_entities = np.concatenate((negative_entities, negative_entities_new))
            negative_entities = np.concatenate((label.reshape([-1]), negative_entities))
            y = np.zeros([max_len], dtype=self.float_dtype)
            y[:len(label)] = 1.
            return y, negative_entities
        y_and_negative_entities = np.asarray(list(map(get,triple,label)))
        y                 = self.float_dtype(y_and_negative_entities[:,0,:])
        negative_entities = self.int_dtype(y_and_negative_entities[:,1,:])
        return y, negative_entities

    def get_label_and_negs_mp(self, triple, label, is_tail_prediction):
        label_lengths           = [len(y) for y in label]
        max_len                 = max(label_lengths)+self.params.ent_x_size
        pool                    = mp.Pool(self.params.num_data_workers)
        fn                      = partial(get_label_and_negs_single, max_len=max_len, ent_vocab_size=self.params.ent_vocab_size, ent_x_size=self.params.ent_x_size, is_tail_prediction=is_tail_prediction, entities = self.entities)
        y_and_negative_entities = np.asarray(list(pool.starmap(fn, zip(triple, label))), dtype=self.int_dtype)
        pool.close()
        y                       = self.float_dtype(y_and_negative_entities[:,0,:])
        negative_entities       = self.int_dtype(y_and_negative_entities[:,1,:])
        return y, negative_entities

def get_negative_entities_single(triple, label, ent_vocab_size, ent_x_size, is_tail_prediction, entities):
    positive_entity     = triple[2] if is_tail_prediction else triple[0]
    mask                = np.ones([ent_vocab_size], dtype=np.bool)
    label               = np.asarray(label, dtype=np.int32)
    mask[label]         = 0
    negative_entities   = np.asarray(np.random.choice(entities[mask], ent_x_size, replace=False)).reshape([-1])
    negative_entities   = np.concatenate((positive_entity.reshape([-1]), negative_entities))
    return negative_entities

def get_label_and_negs_single(triple, label, max_len, ent_vocab_size, ent_x_size, is_tail_prediction, entities):
    negative_length   = max_len - len(label)
    mask              = np.ones([ent_vocab_size], dtype=np.bool)
    label             = np.asarray(label, dtype=np.int32)
    mask[label]       = 0
    negative_entities = np.asarray(np.random.choice(entities[mask], ent_x_size, replace=False)).reshape([-1])
    if negative_length > ent_x_size:
        negative_entities_new = np.random.choice(negative_entities, negative_length-ent_x_size, replace=True).reshape([-1])
        negative_entities = np.concatenate((negative_entities, negative_entities_new))
    negative_entities = np.concatenate((label.reshape([-1]), negative_entities))
    y = np.zeros([max_len], dtype=np.int32)
    y[:len(label)] = 1.
    return y, negative_entities




'''DataPipeline 1TON'''
class DataPipeline_Train_Tail_1toN(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='tail_prediction')

    def get_batch(self, drop_last_batch=True):
        triples                               = self.triples
        np.random.shuffle(triples)
        num_batches                           = int(len(triples)//self.params.batch_size) if drop_last_batch == True else int(math.ceil(len(triples)/self.params.batch_size))
        for i in range(num_batches):
            _startidx                         = i * self.params.batch_size
            batch                             = triples[_startidx : _startidx + self.params.batch_size]
            triple, label, subsampling_weight = zip(*[[self.int_dtype(x['triple']), self.int_dtype(x['label']), self.float_dtype(x['subsampling_weight'])] for x in batch])
            triple, subsampling_weight        = np.asarray(triple, dtype=self.int_dtype), np.asarray(subsampling_weight, dtype=self.float_dtype).reshape([-1])
            h                                 = np.asarray(triple[:,0], dtype=self.int_dtype)
            r                                 = np.asarray(triple[:,1], dtype =self.int_dtype)
            t                                 = np.asarray(triple[:,2], dtype =self.int_dtype)
            y                                 = self.get_label(triple, label, True)
            if self.params.is_label_smoothing == True:
                    y                         = (1.0 - self.params.label_smoothing_epsilon)*y + (1.0/self.params.ent_vocab_size)
            yield (torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(np.zeros([1], dtype=self.int_dtype)), torch.from_numpy(subsampling_weight), 'tail_prediction'), torch.from_numpy(y)

class DataPipeline_Train_Head_1toN(DataPipeline):
    '''This class is not used'''
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='head_prediction')

    def get_batch(self, drop_last_batch=True):
        triples                               = self.triples
        np.random.shuffle(triples)
        num_batches                           = int(len(triples)//self.params.batch_size) if drop_last_batch == True else int(math.ceil(len(triples)/self.params.batch_size))
        for i in range(num_batches):
            _startidx                         = i * self.params.batch_size
            batch                             = triples[_startidx : _startidx + self.params.batch_size]
            triple, label, subsampling_weight = zip(*[[self.int_dtype(x['triple']), self.int_dtype(x['label']), self.float_dtype(x['subsampling_weight'])] for x in batch])
            triple, subsampling_weight        = np.asarray(triple, dtype=self.int_dtype), np.asarray(subsampling_weight, dtype=self.float_dtype).reshape([-1])
            h                                 = np.asarray(triple[:,0], dtype=self.int_dtype)
            r                                 = np.asarray(triple[:,1], dtype =self.int_dtype)
            t                                 = np.asarray(triple[:,2], dtype =self.int_dtype)
            y                                 = self.get_label(triple, label, False)
            if self.params.is_label_smoothing == True:
                    y                    = (1.0 - self.params.label_smoothing_epsilon)*y + (1.0/self.params.ent_vocab_size)
            yield (torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(np.zeros([1], dtype=self.int_dtype)), torch.from_numpy(subsampling_weight), 'head_prediction'), torch.from_numpy(y)

class DataPipeline_Test_Tail_1toN(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='tail_prediction')

    def get_batch(self, drop_last_batch=True):
        triples                          = self.triples
        num_batches                      = int(len(triples)//self.params.eval_batch_size) if drop_last_batch == True else int(math.ceil(len(triples)/self.params.eval_batch_size))
        for i in range(num_batches):
            _startidx                    = i * self.params.eval_batch_size
            batch                        = triples[_startidx : _startidx + self.params.eval_batch_size]
            triple, label                = zip(*[[self.int_dtype(x['triple']),self.int_dtype(x['label'])] for x in batch])
            triple                       = np.asarray(triple, dtype=self.int_dtype)
            h                            = np.asarray(triple[:,0], dtype=self.int_dtype)
            r                            = np.asarray(triple[:,1], dtype =self.int_dtype)
            t                            = np.asarray(triple[:,2], dtype =self.int_dtype)
            y                            = self.get_label(triple, label, True)
            yield (torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(np.zeros([1], dtype=self.int_dtype)), torch.from_numpy(np.zeros([1], dtype=self.float_dtype)), 'tail_prediction'), torch.from_numpy(y).byte()

class DataPipeline_Test_Head_1toN(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='head_prediction')

    def get_batch(self, drop_last_batch=True):
        triples                               = self.triples
        num_batches                           = int(len(triples)//self.params.eval_batch_size) if drop_last_batch == True else int(math.ceil(len(triples)/self.params.eval_batch_size))
        for i in range(num_batches):
            _startidx                         = i * self.params.eval_batch_size
            batch                             = triples[_startidx : _startidx + self.params.eval_batch_size]
            triple, label                     = zip(*[[self.int_dtype(x['triple']), self.int_dtype(x['label'])] for x in batch])
            triple                            = np.asarray(triple, dtype=self.int_dtype)
            h                                 = np.asarray(triple[:,0], dtype=self.int_dtype)
            r                                 = np.asarray(triple[:,1], dtype =self.int_dtype)
            t                                 = np.asarray(triple[:,2], dtype =self.int_dtype)
            y                                 = self.get_label(triple, label, False)
            prediction                        = 'head_prediction'
            if self.params.reverse == True:
                h, t                     = t, h
                r                        = r + self.params.rel_vocab_size_half
                prediction               = 'tail_prediction'
            yield (torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(np.zeros([1], dtype=self.int_dtype)), torch.from_numpy(np.zeros([1], dtype=self.float_dtype)), prediction), torch.from_numpy(y).byte()





'''DataPipeline 1TOX'''
class DataPipeline_Train_Tail_1toX(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='tail_prediction')

    def get_batch(self, drop_last_batch=True):
        triples                               = self.triples
        np.random.shuffle(triples)
        num_batches                           = int(len(triples)//self.params.batch_size) if drop_last_batch == True else int(math.ceil(len(triples)/self.params.batch_size))
        for i in range(num_batches):
            _startidx                         = i * self.params.batch_size
            batch                             = triples[_startidx : _startidx + self.params.batch_size]
            triple, label, subsampling_weight = zip(*[[self.int_dtype(x['triple']), self.int_dtype(x['label']), self.float_dtype(x['subsampling_weight'])] for x in batch])
            triple, subsampling_weight        = np.asarray(triple, dtype=self.int_dtype), np.asarray(subsampling_weight, dtype=self.float_dtype).reshape([-1])
            h                                 = np.asarray(triple[:,0], dtype=self.int_dtype)
            r                                 = np.asarray(triple[:,1], dtype =self.int_dtype)
            t                                 = np.asarray(triple[:,2], dtype =self.int_dtype)
            negative_entities                 = self.get_negative_entities(triple, label, True)
            y                                 = [1]; y.extend([0]*self.params.ent_x_size)
            y                                 = np.tile(np.asarray(y, dtype=self.float_dtype),(len(triple),1))
            if self.params.is_label_smoothing == True:
                    y                         = (1.0 - self.params.label_smoothing_epsilon)*y + (1.0/self.params.ent_vocab_size)
            yield (torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(negative_entities), torch.from_numpy(subsampling_weight), 'tail_prediction'), torch.from_numpy(y)

class DataPipeline_Train_Head_1toX(DataPipeline):
    '''This class is not used'''
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='head_prediction')

    def get_batch(self, drop_last_batch=True):
        triples                               = self.triples
        np.random.shuffle(triples)
        num_batches                           = int(len(triples)//self.params.batch_size) if drop_last_batch == True else int(math.ceil(len(triples)/self.params.batch_size))
        for i in range(num_batches):
            _startidx                         = i * self.params.batch_size
            batch                             = triples[_startidx : _startidx + self.params.batch_size]
            triple, label, subsampling_weight = zip(*[[self.int_dtype(x['triple']), self.int_dtype(x['label']), self.float_dtype(x['subsampling_weight'])] for x in batch])
            triple, subsampling_weight        = np.asarray(triple, dtype=self.int_dtype), np.asarray(subsampling_weight, dtype=self.float_dtype).reshape([-1])
            h                                 = np.asarray(triple[:,0], dtype=self.int_dtype)
            r                                 = np.asarray(triple[:,1], dtype =self.int_dtype)
            t                                 = np.asarray(triple[:,2], dtype =self.int_dtype)
            negative_entities                 = self.get_negative_entities(triple, label, False)
            y                                 = [1]; y.extend([0]*self.params.ent_x_size)
            y                                 = np.tile(np.asarray(y, dtype=self.float_dtype),(len(triple),1))
            if self.params.is_label_smoothing == True:
                    y                         = (1.0 - self.params.label_smoothing_epsilon)*y + (1.0/self.params.ent_vocab_size)
            yield (torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(negative_entities), torch.from_numpy(subsampling_weight), 'head_prediction'), torch.from_numpy(y)





'''DataPipeline 1TONX'''
class DataPipeline_Train_Tail_1toNX(DataPipeline):
    def __init__(self, params, type, data_loader):
        super().__init__(params, type, data_loader, mode='tail_prediction')

    def get_batch(self, drop_last_batch=True):
        triples                               = self.triples
        np.random.shuffle(triples)
        num_batches                           = int(len(triples)//self.params.batch_size) if drop_last_batch == True else int(math.ceil(len(triples)/self.params.batch_size))
        for i in range(num_batches):
            _startidx                         = i * self.params.batch_size
            batch                             = triples[_startidx : _startidx + self.params.batch_size]
            triple, label, subsampling_weight = zip(*[[self.int_dtype(x['triple']), self.int_dtype(x['label']), self.float_dtype(x['subsampling_weight'])] for x in batch])
            triple, subsampling_weight        = np.asarray(triple, dtype=self.int_dtype), np.asarray(subsampling_weight, dtype=self.float_dtype).reshape([-1])
            h                                 = np.asarray(triple[:,0], dtype=self.int_dtype)
            r                                 = np.asarray(triple[:,1], dtype =self.int_dtype)
            t                                 = np.asarray(triple[:,2], dtype =self.int_dtype)
            y, negative_entities              = self.get_label_and_negs(triple, label, True)
            if self.params.is_label_smoothing == True:
                    y                         = (1.0 - self.params.label_smoothing_epsilon)*y + (1.0/self.params.ent_vocab_size)
            yield (torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t), torch.from_numpy(negative_entities), torch.from_numpy(subsampling_weight), 'tail_prediction'), torch.from_numpy(y)





'''BidirectionalOneShotIterator'''
    class BidirectionalOneShotIterator(object):
        def __init__(self, data_iterator_tail, data_iterator_head):
            self.iterator_tail = data_iterator_tail.get_batch()
            self.iterator_head = data_iterator_head.get_batch()
            self.step          = 0

        def get_batch(self):
            self.step += 1
            if self.step % 2 == 0:
                data = next(self.iterator_head)
            else:
                data = next(self.iterator_tail)
            yield data
