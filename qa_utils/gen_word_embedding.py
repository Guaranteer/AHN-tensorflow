import json

def load_json(filename):
    with open(filename) as f1:
        return json.load(f1)

train = load_json('../qg/train_clean1.json')
val = load_json('../qg/val_clean1.json')
test = load_json('../qg/test_clean1.json')

vocal = dict()

for item in train:
    question = item[2].split()
    answer = item[3].split()
    for word in question:
        if word in vocal:
            vocal[word] += 1
        else:
            vocal[word] = 1

    for word in answer:
        if word in vocal:
            vocal[word] += 1
        else:
            vocal[word] = 1

for item in val:
    question = item[2].split()
    answer = item[3].split()
    for word in question:
        if word in vocal:
            vocal[word] += 1
        else:
            vocal[word] = 1

    for word in answer:
        if word in vocal:
            vocal[word] += 1
        else:
            vocal[word] = 1

for item in test:
    question = item[2].split()
    answer = item[3].split()
    for word in question:
        if word in vocal:
            vocal[word] += 1
        else:
            vocal[word] = 1

    for word in answer:
        if word in vocal:
            vocal[word] += 1
        else:
            vocal[word] = 1

print(vocal)
print(len(vocal.keys()))

key_list = [key for key in vocal.keys() if vocal[key] > 1]
key_del = [key for key in vocal.keys() if vocal[key] < 2]
print(key_list)
print(len(key_list))
print(key_del)
print(len(key_del))
print(len(train)+len(val)+len(test))

import gensim
import numpy as np
wv = gensim.models.KeyedVectors.load_word2vec_format('../word2vec/word2vec.bin',binary=True)



embedding = list()
word2index = dict()
index2word = dict()
ind = 2
index2word[0] = '<UNKNOWN>'
index2word[1] = '<PAD>'
word2index['<UNKNOWN>'] = 0
word2index['<PAD>'] = 1
embedding.append(0.6*np.random.rand(300)-0.3)
embedding.append(0.6*np.random.rand(300)-0.3)
for key in vocal.keys():
    if key in wv:
        embedding.append(wv[key])
        word2index[key] = ind
        index2word[ind] = key
        ind += 1
    else:
        print(key)
embedding = np.stack(embedding)
print(embedding)
print(word2index)
print(len(embedding))
print(embedding[10].sum())
print(embedding[1].sum())

import pickle
with open('word2index.pkl','wb') as f:
    pickle.dump(word2index,f)
with open('index2word.pkl','wb') as f:
    pickle.dump(index2word,f)
with open('embedding.pkl', 'wb') as f:
    pickle.dump(embedding, f)








