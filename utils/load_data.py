import os
import numpy as np

class LoadData(object):
    def __init__(self, data_path, dataset_name, dataset_strategy, reverse = True):
        self.data_path               = data_path
        self.dataset_name            = dataset_name
        if dataset_strategy == 'one_to_nx': dataset_strategy   = 'one_to_n'
        self.dataset_strategy        = dataset_strategy
        self.dataset_path            = os.path.join(self.data_path,self.dataset_name,self.dataset_strategy)
        self.reverse                 = True if dataset_strategy=='one_to_n' else reverse

        self.rh2t, self. rt2h        = self._load_multi_label(train=True)
        self.rh2t_all, self.rt2h_all = self._load_multi_label(train=False)
        #self.num_typs, self.relid2typ = self._load_rel_typs()

    def load_train_triples(self, file_numbers=None, mode='tail_prediction'):
        '''
            This function loads triplets specific to a worker.
            Input : data_path="./data/", dataset_name="FB15K-237", file_numbers=[0]
            Output: Triplets as a np array where each item is [h,r,t] with their ids, where [h,r,t] = [head, relation, tail]
            Size is [None,3]
        '''
        triples_tr     = []
        single_file    = True if file_numbers[0]==-1 else False
        reverse        = True if self.reverse == True and self.dataset_strategy=='one_to_x' else False
        rel_vocab_size = self.params['rel_vocab_size_half']

        if single_file==True:
            path = os.path.join(self.dataset_path,'train.txt')
            with open(path, 'rt') as f:
                for line in f.readlines():
                    h, r, t              = [int(x.strip()) for x in line.split()]
                    if self.dataset_strategy=='one_to_x':  subsampling_weight   = len(self.rh2t[r][h]) + len(self.rt2h[r][t])
                    else:
                        if r < self.params['rel_vocab_size_half']:
                            subsampling_weight   = len(self.rh2t[r][h]) + len(self.rt2h[r][t])
                        else:
                             temp_r = r - self.params['rel_vocab_size_half']
                             subsampling_weight   = len(self.rh2t[temp_r][t]) + len(self.rt2h[temp_r][h])
                    subsampling_weight   = np.sqrt(1/subsampling_weight)

                    if mode == 'tail_prediction':
                        if self.params['rel_vocab_size_half'] != 0 and r >= self.params['rel_vocab_size_half']:
                            triples_tr      += [{'triple':(h, r, t), 'label':self.rt2h[r-self.params['rel_vocab_size_half']][h], 'subsampling_weight':subsampling_weight}]
                        else:
                            triples_tr      += [{'triple':(h, r, t), 'label':self.rh2t[r][h], 'subsampling_weight':subsampling_weight}]
                        if reverse == True:
                            triples_tr  += [{'triple':(t, r+rel_vocab_size, h), 'label':self.rt2h[r][t], 'subsampling_weight':subsampling_weight}]
                    else:
                        triples_tr      += [{'triple':(h, r, t), 'label':self.rt2h[r][t], 'subsampling_weight':subsampling_weight}]
        return triples_tr

    def load_test_triples(self,filename,mode='tail_prediction'):
        '''
            This function loads triplets specific to a worker.
            Input : data_path="./data/", dataset_name="FB15K-237"
                    filename= train or test
            Output: Triplets as a np array where each item is [h,r,t] with their ids, where [h,r,t] = [head, relation, tail]
            Size is [None,3]
        '''
        path = os.path.join(self.dataset_path, filename+'.txt')
        triples_te = []
        with open(path, 'rt') as f:
            for line in f.readlines():
                h, r, t     = [int(x.strip()) for x in line.split()]
                if mode == 'tail_prediction':
                    triples_te  += [{'triple':(h, r, t), 'label':self.rh2t_all[r][h]}]
                else:
                    triples_te  += [{'triple':(h, r, t), 'label':self.rt2h_all[r][t]}]
        return triples_te

    def load_vocab_size(self):
        '''
            This function returns global entity vocab size and relation vocab size.
            Input : data_path="./data/", dataset_name="FB15K-237"
            Output: params dictionary with ent_vocab_size and rel_vocab_size
        '''
        entity_relation_size_path = os.path.join(self.dataset_path, 'entity_relation_size.txt')
        self.params = {}
        with open(entity_relation_size_path, 'rt') as f:
            line = f.readline()
            ent_vocab_size, rel_vocab_size         = [int(x.strip()) for x in line.split()]
            self.params['ent_vocab_size']          = ent_vocab_size
            if self.reverse == True:
                self.params['rel_vocab_size']      = rel_vocab_size * 2
                self.params['rel_vocab_size_half'] = rel_vocab_size
            else:
                self.params['rel_vocab_size']      = rel_vocab_size
                self.params['rel_vocab_size_half'] = 0
        return self.params

    def __load_multi_label(self, rh2t_path, rt2h_path):
        rh2t = {}
        rt2h = {}
        with open(rh2t_path, 'rt') as f:
            for line in f.readlines():
                r, h, t        = line.split("\t")
                r, h           = int(r), int(h)
                if r not in rh2t:
                    rh2t[r]    = {}
                if h not in rh2t[r]:
                    rh2t[r][h] = []
                rh2t[r][h].extend([int(x.strip()) for x in t.split()])
        with open(rt2h_path, 'rt') as f:
            for line in f.readlines():
                r, t, h        = line.split("\t")
                r, t           = int(r), int(t)
                if r not in rt2h:
                    rt2h[r]    = {}
                if t not in rt2h[r]:
                    rt2h[r][t] = []
                rt2h[r][t].extend([int(x.strip()) for x in h.split()])
        return rh2t, rt2h

    def _load_rel_typs(self):
        filename = os.path.join(self.dataset_path, 'relid2type.txt')
        rel_typs = {}
        width = 0
        all_typs = set()
        if not os.path.exists(filename):
            return 0, rel_typs
        with open(filename, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line:
                    xs = line.split('\t')
                    cur_typs = []
                    for x in xs[1:]:
                        cur_typ = list(map(int, x.split(' ')))
                        all_typs.update(cur_typ)
                        # cur_typs.append(cur_typ)
                        cur_typs.extend(cur_typ)
                    rel_typs[int(xs[0])] = cur_typs
                    width = max(width, len(cur_typs))
        num_typs = len(all_typs)
        max_typ = max(all_typs)+1
        relid2typ = np.zeros((len(rel_typs), width), dtype=np.int64)
        for relid, typ in rel_typs.items():
            typ = typ+ [max_typ for i in range(width-len(typ))]
            relid2typ[relid,:] = typ
        return num_typs, relid2typ

    def _load_multi_label(self, train=True):
        '''
            This function returns rh2t anf rt2h mappings in training triples set.
            Input : data_path="./data/", dataset_name="FB15K-237"
            Output: rt2h and rh2t both are in format as rh2t[relation_id][head_entity_id] = [list of tail ids]
        '''
        type = 'train' if train==True else 'all'
        rh2t_path  = os.path.join(self.dataset_path,'rh2t.txt.'+type)
        rt2h_path  = os.path.join(self.dataset_path,'rt2h.txt.'+type)
        rh2t, rt2h = self.__load_multi_label(rh2t_path, rt2h_path)
        return rh2t, rt2h

    def load_adjacency_list(self, mode='tail_prediction'):
        '''
            This function handles the case for reverse of one_to_x and one_to_n and not head prediction of one_to_x
        '''
        triples_tr     = []
        reverse        = True if self.reverse == True and self.dataset_strategy=='one_to_x' else False
        rel_vocab_size = self.params['rel_vocab_size_half']

        edge_index, edge_type = [], []

        path = os.path.join(self.dataset_path,'train.txt')
        with open(path, 'rt') as f:
            for line in f.readlines():
                h, r, t              = [int(x.strip()) for x in line.split()]
                if mode == 'tail_prediction':
                    if self.params['rel_vocab_size_half'] != 0 and r >= self.params['rel_vocab_size_half']:
                        edge_index.append((h,t))
                        edge_type.append(r)
                    if reverse == True:
                        edge_index.append((t,h))
                        edge_type.append(r+rel_vocab_size)
                else:
                    print("Danger Code")
                    raise NotImplementedError
        return edge_index, edge_type
