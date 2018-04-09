import json
import pickle
import numpy as np
import random
import os
import h5py
import gensim


def load_file(filename):
    return pickle.load(open(filename))


class Batcher(object):
    def __init__(self, params, data_file, mode, reshuffle=False):
        # general
        self.reshuffle = reshuffle
        self.max_batch_size = params['batch_size']
        self.data_file = data_file
        self.next_idx = 0
        self.feature_path = params['feature_path']
        self.params = params
        # dataset

        self.wv = gensim.models.KeyedVectors.load_word2vec_format(params['word2vec'],binary=True)
        self.ans_set = self.get_answer()        
        
        self.keys = self.get_keys()
        self.qa_pair = self.get_qa_pair()
        print(len(self.qa_pair))
        # print(self.qa_pair)
        # print(self.keys)


        # frame / motion / question
        self.max_n_frames = params['max_n_frames']
        self.v_dim = params['input_video_dim']
        self.max_q_words = params['max_n_q_words']
        self.q_dim = params['input_ques_dim']

        # load data & initialization
        self.data_index = list(range(len(self.qa_pair)))
        print(len(self.data_index))
        if self.reshuffle:
            random.shuffle(self.data_index)

    def reset(self):
        # reset for next epoch
        self.next_idx = 0
        if self.reshuffle:
            random.shuffle(self.data_index)

    def generate(self):
        # generate w2v question vectors

        # next batch size
        batch_size = self.max_batch_size

        # initialization
        img_frame_vecs = np.zeros((batch_size, self.max_n_frames, self.v_dim), dtype=float)
        img_frame_n = np.zeros((batch_size), dtype=int)
        ques_vecs = np.zeros((batch_size, self.max_q_words, self.q_dim), dtype=float)
        ques_n = np.zeros((batch_size), dtype=int)
        ans_vec = np.zeros((batch_size), dtype=int)
        type_vec = np.zeros((batch_size), dtype=int)


        for i in range(batch_size):
            curr_data_index = self.data_index[self.next_idx]
            vid = self.qa_pair[curr_data_index][0]
            self.next_idx += 1
            if self.next_idx >= len(self.data_index):
                self.next_idx -= random.randint(1,len(self.data_index))

            if not os.path.exists(self.feature_path + '/feat/%s.h5' % vid) :
                continue
            with h5py.File(self.feature_path + '/feat/%s.h5' % vid, 'r') as hf:
                fg = np.asarray(hf['fg'])
                bg = np.asarray(hf['bg'])
                feat = np.hstack([fg, bg])
            with h5py.File(self.feature_path + '/flow/%s.h5' % vid, 'r') as hf:
                fg2 = np.asarray(hf['fg'])
                bg2 = np.asarray(hf['bg'])
                feat2 = np.hstack([fg2, bg2])
            feat = feat + feat2

            inds = np.floor(np.arange(0,len(feat)-0.1,len(feat)/self.params["max_n_frames"])).astype(int)
            frames = feat[inds,:]
            frames = np.vstack(frames)
            if len(frames) > self.max_n_frames:
                frames = frames[:self.max_n_frames]
            n_frames = len(frames)
            img_frame_vecs[i, :n_frames, :] = frames
            img_frame_n[i] = n_frames

            # acquire question vector, answer id and type id
            tmp_ques_vecs, tmp_ans, tmp_ty = self.get_qa_vec(curr_data_index)

            tmp_ques_vecs = tmp_ques_vecs[tmp_ques_vecs.sum(axis=1) != 0]  # there may be words not in pretrained models represented as all-zero vectors
            ques_vecs[i, :tmp_ques_vecs.shape[0], :] = tmp_ques_vecs
            ques_n[i] = tmp_ques_vecs.shape[0]

            ans_vec[i] = tmp_ans
            type_vec[i] = tmp_ty




        return img_frame_vecs, img_frame_n, ques_vecs, ques_n, ans_vec, type_vec


    def get_answer(self):
        with open(self.params['answer'],'r') as fr:
            data = json.load(fr)
        ans_set = list()
        for key,item in data.items():
            ans_set.extend(item)
        return ans_set

    def get_keys(self):
        with open(self.data_file,'r') as fr:
            data = json.load(fr)
        keys = list()
        for key, item in data.items():
            keys.append(key[2:])
        return keys


    def get_qa_pair(self):
        with open(self.params['question'],'r') as f:
            data = json.load(f)

        qa_pair = list()
        for key,items in data.items():
            if key[2:] in self.keys:
                for item in items:
                    item.insert(0,key[2:])
                    qa_pair.append(item)

        return  qa_pair


    # def get_qa_pair(self):
    #     with open(self.params['question'],'r') as f:
    #         data = json.load(f)
    #     qa_pair = dict()
    #     for key,item in data.items():
    #         qa_pair[key[2:]] = item
    #     return  qa_pair

    def get_qa_vec(self,vid):
        data = self.qa_pair[vid]
        ty = data[1]
        ques = data[2]
        ans = data[3]

        q_words = ques.split()
        ques_vec = []
        for word in q_words:
            if word in self.wv:
                vec = self.wv[word]
            else:
                # print('word %s not in vocab' % word)
                vec = np.zeros(shape=(300))
            ques_vec.append(vec)
        ques_vec = np.vstack(ques_vec)

        if ans in self.ans_set:
            ans_ind = self.ans_set.index(ans)
        else:
            print('answer not in set!!!!!!')
            ans_ind = 0

        return  ques_vec,ans_ind,ty