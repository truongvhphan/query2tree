import pickle

import numpy as np

path_p = 'nell_bert_pin.pkl'

with open(path_p, 'rb') as f:
    data = pickle.load(f)

print(data)

def convert2id(ls):
    rs = []
    for i in ls:
        try:
            rs.append(int(i))
        except:
            continue
    return rs

mrr = []
hitk = []
l = len(data)
for d in data:

    h = d[0]
    truth = sorted(list(map(int, d[-2])))
    pred = sorted(convert2id(d[-1]))
    rr = []
    hit = []
    for a in truth:
        try:
            rs = pred.index(a)
            print(rs, '---', 1/(rs+1))
            rr.append(1/(rs+1))
            if rs < 3:
                hit.append(rs+1)
        except:
            continue
    mrr.append(np.sum(rr))
    hitk.append(np.sum(hit))

print(f'MRR: {np.sum(mrr)/l}, number of data: {l}')
print(f'Hits@3: {np.sum(hitk)/l}, number of data: {len(data)}')
