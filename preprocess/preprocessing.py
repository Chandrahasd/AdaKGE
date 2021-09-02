import argparse
import sys
import os

def read_triples(path):
    triples = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            h, r, t = line.split()
            triples += [(h.strip(), r.strip(), t.strip())]
    return triples

def load_triple(dataset_path):
    triples_tr = read_triples(dataset_path+'pre/'+'train.txt')
    triples_va = read_triples(dataset_path+'pre/'+'valid.txt')
    triples_te = read_triples(dataset_path+'pre/'+'test.txt')
    triples_all = triples_tr + triples_va + triples_te
    return triples_all, triples_tr, triples_va, triples_te

def write_triples(path, e2idx, r2idx, triples):
    with open(path, 'wt') as f:
        for (_h, _r, _t) in triples:
            h, r, t = e2idx[_h], r2idx[_r], e2idx[_t]
            f.write(str(h)+" "+str(r)+" "+str(t))
            f.write("\n")

def write_one_to_n_train_triple(path, e2idx, r2idx, triples):
    rh2t = {}
    num_rel = len(r2idx)
    for (_h, _r, _t) in triples:
        h, r, t = e2idx[_h], r2idx[_r], e2idx[_t]
        if (r,h) not in rh2t:
            rh2t[(r,h)] = t
        if (r+num_rel,t) not in rh2t:
            rh2t[(r+num_rel,t)] = h
    with open(path, 'wt') as f:
        for key,value in rh2t.items():
            f.write(str(key[1])+"\t"+str(key[0])+"\t"+str(value))
            f.write("\n")

def save_triple(dataset_path, e2idx, r2idx, triples_tr, triples_va, triples_te, one_to_x=False):
    if one_to_x == True:
        write_triples(dataset_path+'train.txt', e2idx, r2idx, triples_tr)
    else:
        write_one_to_n_train_triple(dataset_path+'train.txt', e2idx, r2idx, triples_tr)
    write_triples(dataset_path+'valid.txt', e2idx, r2idx, triples_va)
    write_triples(dataset_path+'test.txt', e2idx, r2idx, triples_te)

def build_vocab(triples):
    params = {}
    e_set = {h for (h, r, t) in triples} | {t for (h, r, t) in triples}
    r_set = {r for (h, r, t) in triples}
    params['ent_vocab_size'] = len(e_set)
    params['rel_vocab_size'] = len(r_set)
    e2idx = {e: idx for idx, e in enumerate(sorted(e_set))}
    r2idx = {r: idx for idx, r in enumerate(sorted(r_set))}
    return e2idx, r2idx, params

def save_vocab(dataset_path, params):
    entity_relation_size_path = dataset_path + 'entity_relation_size.txt'
    with open(entity_relation_size_path, 'wt') as f:
        f.write(str(params['ent_vocab_size'])+" "+str(params['rel_vocab_size']))

def save_names(dataset_path, name2id, filename='name2id.txt'):
    lines = []
    for name, nid in name2id.items():
        lines.append("{name}\t{nid}".format(name=name, nid=nid))
    with open(os.path.join(dataset_path, filename), 'w') as fout:
        fout.write("\n".join(lines))

def save_relation_vocab(dataset_path, r2id):
    lines = []
    typ_vocab = {}
    cur_id = [0]
    typ2id = {}
    def getId(typ):
        if typ not in typ2id:
            typ2id[typ] = str(cur_id[0])
            cur_id[0] = cur_id[0]+1
        return typ2id[typ]
    for rels, rid in r2id.items():
        typs = []
        for rel in rels.strip().split('.'):
            cur_typs = []
            for typ in rel.strip().split('/')[1:]:
                cur_typs.append(getId(typ))
            typs.append(" ".join(cur_typs))
        cur_line = "\t".join([str(rid)]+typs)
        lines.append(cur_line)
    with open(os.path.join(dataset_path, 'relid2type.txt'), 'w') as fout:
        fout.write("\n".join(lines))
    lines = []
    for typ, tid in typ2id.items():
        lines.append("\t".join([typ, tid]))
    with open(os.path.join(dataset_path, 'reltype2id.txt'), 'w') as fout:
        fout.write("\n".join(lines))

def save_multi_label(triples, e2idx, r2idx, rh2t_path, rt2h_path):
    rh2t = {}
    rt2h = {}
    for (_h, _r, _t) in triples:
        h, r, t = e2idx[_h], r2idx[_r], e2idx[_t]
        if (r,h) not in rh2t:
            rh2t[(r,h)] = [t]
        else:
            if t not in rh2t[(r,h)]:
                rh2t[(r,h)].append(t)
        if (r,t) not in rt2h:
            rt2h[(r,t)] = [h]
        else:
            if h not in rt2h[(r,t)]:
                rt2h[(r,t)].append(h)
    with open(rh2t_path, 'wt') as f:
        for key,value in rh2t.items():
            tail_list_string = ' '.join(map(str,value))
            f.write(str(key[0])+"\t"+str(key[1])+"\t"+tail_list_string)
            f.write("\n")
    with open(rt2h_path, 'wt') as f:
        for key,value in rt2h.items():
            head_list_string = ' '.join(map(str,value))
            f.write(str(key[0])+"\t"+str(key[1])+"\t"+head_list_string)
            f.write("\n")

def save_partitions(dataset_path, partitions):
    base_file_path = dataset_path+'train.txt.'
    file = []
    for i in range(partitions):
        f = open(base_file_path+str(i),'wt')
        file.append(f)
    i = 0
    with open(dataset_path+'train.txt', 'rt') as f:
        for line in f.readlines():
            file[i].write(line)
            i = (i+1)%partitions
    for i in range(partitions):
        file[i].close()

def main(data_path,dataset_name,partitions):
    base_dataset_path = data_path+dataset_name+"/"
    triples_all, triples_tr, triples_va, triples_te = load_triple(base_dataset_path)
    e2idx, r2idx, params = build_vocab(triples_all)

    dataset_path = base_dataset_path + 'one_to_x/'
    os.mkdir(dataset_path)
    save_triple(dataset_path, e2idx, r2idx, triples_tr, triples_va, triples_te, one_to_x=True)
    save_vocab(dataset_path, params)
    save_names(dataset_path, e2idx, 'entity2id.txt')
    save_names(dataset_path, r2idx, 'relation2id.txt')
    save_relation_vocab(dataset_path, r2idx)
    save_multi_label(triples_tr, e2idx, r2idx, dataset_path+"rh2t.txt.train",dataset_path+"rt2h.txt.train")
    save_multi_label(triples_all, e2idx, r2idx, dataset_path+"rh2t.txt.all",dataset_path+"rt2h.txt.all")
    # save_partitions(dataset_path, partitions)

    dataset_path = base_dataset_path + 'one_to_n/'
    os.mkdir(dataset_path)
    save_triple(dataset_path, e2idx, r2idx, triples_tr, triples_va, triples_te, one_to_x=False)
    save_vocab(dataset_path, params)
    save_names(dataset_path, e2idx, 'entity2id.txt')
    save_names(dataset_path, r2idx, 'relation2id.txt')
    save_relation_vocab(dataset_path, r2idx)
    save_multi_label(triples_tr, e2idx, r2idx, dataset_path+"rh2t.txt.train",dataset_path+"rt2h.txt.train")
    save_multi_label(triples_all, e2idx, r2idx, dataset_path+"rh2t.txt.all",dataset_path+"rt2h.txt.all")
    # save_partitions(dataset_path, partitions)

def getParser():
    parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str,  default="./../../data/", help="Required, Base path for data folder")
    parser.add_argument("--dataset_name", type=str,  default="yago3-10", help="Required, Dataset Name")
    parser.add_argument("--partitions", type=int,  default=8, help="Required, No. of partitions")
    return parser

if __name__=="__main__":
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        sys.exit(1)
    main(args.data_path, args.dataset_name, args.partitions)
