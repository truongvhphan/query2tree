import pickle

import numpy as np

path_p = 'fb15k_bert_pin.pkl'

with open(path_p, 'rb') as f:
    data = pickle.load(f)

# print(data)

ent2id_path = '../FB15K/entity2id.txt'

ent2id = {}
with open(ent2id_path, 'r', encoding='utf-8') as f:
  data1 = f.readlines()
  # i = 0
  for d in data1[1:]:
    item = d.strip('\n').split('\t')
    ent2id[item[0]] = int(item[1])
    # id2ent[i] = item
    # i += 1

def convert2int(a):
    ls = []
    for j in a:
        try:
            ls.append(ent2id[j])
        except:
            continue
    return ls

mrr = []
hitk = []
l = len(data)
for d in data:
    h = d[0]
    truth = sorted(d[-2])
    # pred = sorted(d[-1])
    pred = sorted(convert2int(d[-1]))
    rr = []
    hit = []
    for a in truth:
        try:
            rs = pred.index(a)
            print(rs, '---', 1/(rs+1))
            rr.append(1/(rs+1))
            if rs < 10:
                hit.append(rs+1)
        except:
            continue
    mrr.append(np.sum(rr))
    hitk.append(np.sum(hit))

print(f'MRR: {np.mean(mrr)}, number of data: {len(data)}')
print(f'Hits@3: {np.sum(hitk)/l}, number of data: {len(data)}')
