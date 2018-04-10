import json
import pickle as pkl
import numpy as np
import random
import os
import h5py

stopwords = ['.','?']

def load_file(filename):
    with open(filename,'rb') as f1:
        return pkl.load(f1)

def load_json(filename):
    with open(filename) as f1:
        return json.load(f1)

class Batcher(object):
    def __init__(self, params, data_file, mode, reshuffle=False):
        # general
        self.reshuffle = reshuffle
        self.next_idx = 0
        self.feature_path = params['feature_path']
        self.params = params
        self.max_batch_size = params['batch_size']

        # dataset
        self.all_word_vec = load_file(params['word_embedding'])
        self.word_dict = load_file(params['word2index'])
        self.key_file = load_json(data_file)

        # frame / motion / question

        self.max_n_frames = params['max_n_frames']
        self.v_dim = params['input_video_dim']
        self.max_q_words = params['max_n_q_words']
        self.max_a_words = params['max_n_a_words']
        self.q_dim = params['input_ques_dim']

        # load data & initialization
        self.data_index = list(range(len(self.key_file)))
        if self.reshuffle:
            random.shuffle(self.data_index)

    def reset(self):
        # reset for next epoch
        self.next_idx = 0
        if self.reshuffle:
            random.shuffle(self.data_index)

    def generate(self):
        # next batch size
        while True:
            batch_size = min(self.max_batch_size, len(self.key_file) - self.next_idx)
            if batch_size <= 0:
                yield None, None, None, None, None, None, None, None, None, None

            img_frame_vecs = np.zeros((batch_size, self.max_n_frames, self.v_dim), dtype=float)
            img_frame_n = np.zeros((batch_size), dtype=int)

            ques_vecs = np.zeros((batch_size, self.max_q_words, self.q_dim), dtype=float)
            ques_word = np.ones((batch_size, self.max_q_words), dtype=int)
            ques_n = np.zeros((batch_size), dtype=int)

            ans_vecs = np.zeros((batch_size, self.max_a_words, self.q_dim), dtype=float)
            ans_word = np.ones((batch_size, self.max_a_words), dtype=int)
            ans_n = np.zeros((batch_size), dtype=int)

            type_vec = np.zeros((batch_size), dtype=int)

            for i in range(batch_size):
                curr_data_index = self.data_index[self.next_idx + i]
                vid = self.key_file[curr_data_index][0][2:].encode('utf-8')

                if not os.path.exists(self.feature_path + '/feat/%s.h5' % vid):
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

                inds = np.floor(np.arange(0, len(feat) - 0.1, len(feat) / self.params["max_n_frames"])).astype(int)
                frames = feat[inds, :]
                frames = np.vstack(frames)
                if len(frames) > self.max_n_frames:
                    frames = frames[:self.max_n_frames]
                n_frames = len(frames)
                img_frame_vecs[i, :n_frames, :] = frames
                img_frame_n[i] = n_frames

                # question
                ques = self.key_file[curr_data_index][2].encode('utf-8').split()
                ques = [word.lower() for word in ques if word not in stopwords and word != '']
                ques = [self.word_dict[word] if word in self.word_dict else 0 for word in ques]
                vector = self.all_word_vec[ques]
                ques_n[i] = min(len(ques), self.max_q_words)
                ques_word[i, :ques_n[i]] = ques[:ques_n[i]]
                ques_vecs[i, :ques_n[i], :] = vector[:ques_n[i], :]

                # answer
                ans = self.key_file[curr_data_index][3].encode('utf-8').split()
                ans = [word.lower() for word in ans if word not in stopwords and word != '' and word != 'EOS']
                ans += ['EOS']
                ans = [self.word_dict[word] if word in self.word_dict else 0 for word in ans]
                if len(ans) > self.max_a_words:
                    ans = ans[:self.max_a_words-1] +  ans[-1:]
                vector = self.all_word_vec[ans]
                ans_n[i] = min(len(ans), self.max_a_words)
                ans_word[i, :ans_n[i]] = ans[:ans_n[i]]
                ans_vecs[i, :ans_n[i], :] = vector[:ans_n[i], :]

                type_vec[i] = self.key_file[curr_data_index][4]

            self.next_idx += batch_size
            yield img_frame_vecs, img_frame_n, ques_vecs, ques_n, ques_word, ans_vecs, ans_n, ans_word, type_vec, batch_size
