# -*- coding: utf-8 -*-
"""cd3_embedding_datalog.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dT135oSKuFATND9tqJH7NYM4arUpAKKO
"""

# !pip install node2vec
import csv
import time
import numpy as np
from gensim.models import Word2Vec
import networkx as nx
from node2vec import Node2Vec
import pickle
from gensim.models import Word2Vec
# from rank_eval import Qrels, Run, evaluate


path = 'fb15k/fb_graph.pkl'
ent2id_path = 'data/FB15K/entity2id.txt'
relation2id_path = 'data/FB15K/relation2id.txt'
train_path = 'data/FB15K/train.tsv'
dev_path = 'data/FB15K/dev.tsv'
test_path = 'data/FB15K/test.tsv'
ent_type_path = 'fb13/entities_types.txt'
tripleid_path = 'fb13/triple_id.txt'
rel2id_path = 'data/FB15K/relation2id.txt'
graph_path = 'fb13/train.tsv'
embedding_transe = 'fb15k/entity2vec64_transe.unif'


triples = []
ent2id = []
rel2id = []
ent_type = []

ent2id = {}
rel2id = {}
id2ent = {}
id2rel = {}

with open(ent2id_path, 'r', encoding='utf-8') as f:
  data = f.readlines()
  for d in data[1:]:
    item = d.strip('\n').split('\t')
    ent2id[item[0]] = item[1]
    id2ent[item[1]] = item[0]

# print(ent2id)
# print(id2ent)
#
with open(rel2id_path, 'r', encoding='utf-8') as f:
  data = f.readlines()
  for d in data[1:]:
    item = d.strip('\n').split('\t')
    rel2id[item[0]] = item[1]
    id2rel[item[1]] = item[0]

# print(rel2id)


# with open(graph_path, 'r', encoding='utf-8') as f:
#   data = csv.reader(f, delimiter='\t')
#   k = open(tripleid_path, 'w', encoding='utf-8')
#
#   for t in data:
#     k.write('{}#{}#{}\n'.format(ent2id[t[0]], rel2id[t[1]], ent2id[t[2]]))
#   k.close()


# Create a graph
# G = nx.Graph()
#
# with open(train_path, 'r', encoding='utf-8') as f:
#   data = csv.reader(f, delimiter='\t')
  # Add edges to the graph
  # for t in data:
  #   head, rel, tail = t[0], t[1], t[2]
  #   G.add_edge(head, tail, label=rel)

# with open(dev_path, 'r', encoding='utf-8') as f:
#   data = csv.reader(f, delimiter='\t')
  # Add edges to the graph
  # for t in data:
  #   head, rel, tail = t[0], t[1], t[2]
  #   G.add_edge(head, tail, label=rel)

# with open(test_path, 'r', encoding='utf-8') as f:
#   data = csv.reader(f, delimiter='\t')
  # Add edges to the graph
  # for t in data:
  #   head, rel, tail = t[0], t[1], t[2]
  #   G.add_edge(head, tail, label=rel)

#
# import pickle
# Save the knowledge graph to disk
# with open('data/FB15k_query2tree/fb15k_kg.pkl', 'wb') as file:
#     pickle.dump(G, file)


#load KG from disk
print("Load the KG from disk")
with open('data/FB15k_query2tree/fb15k_kg.pkl', 'rb') as file:
    G = pickle.load(file)

print(G)
#create sub_KG
# relations = np.unique(list(nx.get_edge_attributes(G,'label').values()))
#
# print(relations)
# s = time.time()
# for r in relations:
#     tmp_g = nx.Graph()
#     for k, v in nx.get_edge_attributes(G,'label').items():
#         print(k, '----', v, '---', r)
#         if v == r:
#             tmp_g.add_edge(k[0], k[1])
#     with open(f'data/FB15k_query2tree/fb15k_relation_{rel2id[r]}.pkl', 'wb') as file:
#         pickle.dump(tmp_g, file)
# ## end sub_KG
# #
# # Generate node embeddings using node2vec
# node2vec = Node2Vec(G, dimensions=384, walk_length=5, num_walks=20, p=0.5, q=2, workers=8)
# model = node2vec.fit(window=10, min_count=1, batch_words=4)
#
#   # Get the node embeddings
# node_embeddings = model.wv
# model.wv.save_word2vec_format(f'data/FB15k_query2tree/fb15k_kg_node2vec.wv')
#
#   # Save model for later use
# model.save(f'data/FB15k_query2tree/fb15k_kg_node2vec.model')
# e = time.time()
# print(f'Training time {e-s}')
## END



#train node2vec and save models
# degrees = G.degree()
# NodeDegree=[]
# NodeName=[]
# # Print the node degrees
# for node, degree in degrees:
#     print(f"Node {node}: Degree {degree}")
#     NodeDegree.append(degree)
#     NodeName.append(node)
# s = time.time()
# # Generate node embeddings using node2vec
# node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, p=0.5, q=2, workers=12)
# model = node2vec.fit(window=10, min_count=1, batch_words=4)
#
# # Get the node embeddings
# node_embeddings = model.wv
# e = time.time()
# print(f'Training time {e-s}')
# # Print the embeddings for each node
# # for node, embedding in zip(node_embeddings.index2entity, node_embeddings.vectors):
# # for node, embedding in zip(node_embeddings.index_to_key, node_embeddings.vectors):
# #     print(f"Node: {node}, Embedding: {embedding}")
#
# # Save embeddings for later use
# model.wv.save_word2vec_format('fb13/fb_kg_node2vec.wv')
#
# # Save model for later use
# model.save('fb13/fb_kg_node2vec.model')
# s_t = time.time()
# print(f'Saving time {s_t - e}')



#load node2vec models

m = Word2Vec.load('data/FB15k_query2tree/fb15k_kg_node2vec.model')
node_embeddings = m.wv

#load TransE models

# embeddings = np.loadtxt(embedding_transe, delimiter=' ')


# Print the embeddings for each node
# for node, embedding in zip(node_embeddings.index2entity, node_embeddings.vectors):
# for node, embedding in zip(node_embeddings.index_to_key, node_embeddings.vectors):
#     print(f"Node: {node}, Embedding: {embedding}")

#create kdtree for each subgraph based on relations
# import pickle
# from scipy.spatial import KDTree
# from sklearn.neighbors import KDTree
#
# relations = np.unique(list(nx.get_edge_attributes(G, 'label').values())).tolist()
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
# data = []
# for node in G.nodes:
    # print(f'Node:{node}, Embedding: {node_embeddings.get_vector(node)})')
    # print(f'Node:{node}, Len: {len(node_embeddings.get_vector(node))}')
    # embedding = node_embeddings.get_vector(node)
    # data.append(embedding)

# neigh = NearestNeighbors(algorithm='ball_tree')
# nn = neigh.fit(data)
# with open(f'data/FB15k_query2tree/fb15k_brute.pkl', 'wb') as kdfile:
#       pickle.dump(nn,kdfile)
# kdtree = KDTree(data)
# with open(f'data/FB15k_query2tree/fb15k_kd_tree.pkl', 'wb') as kdfile:
#       pickle.dump(kdtree,kdfile)
# print('DONE')
## End kdtree for each subgraph based on relations



print("Load the kdtree from disk")
# with open('data/FB15k_query2tree/fb15k_kd_tree.pkl', 'rb') as file:
#     kdtree = pickle.load(file)

with open('data/FB15k_query2tree/fb15k_brute.pkl', 'rb') as file:
    nn = pickle.load(file)

# for index, node in enumerate(G.nodes):
#     print(index, '----', node)
#     id2ent[index] = node

for node in G.nodes:
    degree = G.degree[node]
    print(f'Node: {node} --- Degree : {degree}')
    rs = nn.kneighbors(np.reshape(node_embeddings.get_vector(node), (-1,384)), n_neighbors=degree, return_distance=False)
    print(list(rs[0]))

    # dist, ind = kdtree.query(np.reshape(node_embeddings.get_vector(node), (-1,384)), k=degree)
#     print(ind)
#
#     for i in ind.tolist()[0][1:]:
#         print(id2ent[str(i)])
    break

# '''
q2b_path = 'Q2Bdata/FB15k/test_ans_3i.pkl'
with open(q2b_path, 'rb') as f:
    test_1_p = pickle.load(f)
mrr = []
one_project = []
for k, v in test_1_p.items():
    # print(k, '----', v)
    # h = k[0][0]
    h = k[0][1][0]
    # # h0 = k[0][1][0]
    h1 = k[1][1][0]
    h2 = k[2][1][0]
    ent = id2ent[str(h)]
    # # ent0 = id2ent[str(h0)]
    ent1 = id2ent[str(h1)]
    ent2 = id2ent[str(h2)]
    deg = G.degree[str(ent)]
    # # deg0 = G.degree[str(ent0)]
    deg1 = G.degree[str(ent1)]
    deg2 = G.degree[str(ent2)]
    # # # _, ind = kdtree.query(np.reshape(node_embeddings.get_vector(ent), (-1, 384)), k=deg)
    ind = nn.kneighbors(np.reshape(node_embeddings.get_vector(ent), (-1,384)), n_neighbors=deg, return_distance=False)
    # # ind0 = nn.kneighbors(np.reshape(node_embeddings.get_vector(ent0), (-1, 384)), n_neighbors=deg0, return_distance=False)
    ind1 = nn.kneighbors(np.reshape(node_embeddings.get_vector(ent1), (-1, 384)), n_neighbors=deg1, return_distance=False)
    ind2 = nn.kneighbors(np.reshape(node_embeddings.get_vector(ent2), (-1,384)), n_neighbors=deg2, return_distance=False)
    # #
    #
    tree0 = ind.tolist()[0][1:]
    # # tree_0 = ind0.tolist()[0][1:]
    tree1= ind1.tolist()[0][1:]
    tree2 = ind2.tolist()[0][1:]
    # # tree_1_2 = set(tree1).union(set(tree2))
    # # tree_0_1 = set(tree_0).union(tree_1_2)
    tree = set(tree0).intersection(set(tree1).intersection(set(tree2)))
    # # print(tree)
    rr = []
    for a in list(v):
        try:
            rs = list(tree).index(a)
            print(rs)
            if [h, h1, h2, list(v), list(tree)] not in one_project:
                one_project.append([h, h1, h2, list(v), list(tree)])
            rr.append(1/(rs+1))
        except:
            continue
    if len(rr) > 0:
        mrr.append(np.mean(rr))
    # print(h, h1,list(v), list(tree))

print(f'MRR: {np.mean(mrr)}')
with open('data/FB15k_query2tree/fb15k_3i.pkl', 'wb') as f:
    pickle.dump(one_project, f)
# # print(one_project)
print(len(one_project))
# '''