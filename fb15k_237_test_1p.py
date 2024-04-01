import pickle

import numpy as np

path_p = 'fb15k_237_2in.pkl'
ent2id_path = '../FB15k-237/entities.txt'

ent2id = {}
with open(ent2id_path, 'r', encoding='utf-8') as f:
  data = f.readlines()
  i = 0
  for d in data:
    item = d.strip('\n')
    ent2id[item] = i
    # id2ent[i] = item
    i += 1
    # ent_v.append(item)

print(ent2id)

with open(path_p, 'rb') as f:
    data = pickle.load(f)

print(data[0][2])

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
    truth = sorted(d[1])
    # pred = sorted(convert2int(d[2]))
    pred = sorted(d[2])
    rr = []
    hit = []
    for a in truth:
        try:
            rs = pred.index(a)
            print(rs, '---', 1/(rs+1))
            rr.append(1/(rs+1))
            if rs < 3:
                hit.append((rs+1))
        except:
            continue
    mrr.append(np.max(rr))
    hitk.append(np.sum(hit))

print(f'MRR: {np.mean(mrr)}, number of data: {len(data)}')
print(f'Hits@3: {np.sum(hitk)/l}, number of data: {len(data)}')
