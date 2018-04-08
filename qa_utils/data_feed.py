import json
import pickle
import numpy as np
import random
import os
import h5py
from tsn import extract
import gensim


def load_file(filename):
    return pickle.load(open(filename))


class Batcher(object):
    def __init__(self, params, data_file, mode, reshuffle=False):
        # general
        self.reshuffle = reshuffle
        self.dataset = params['dataset']  # yqf / xhy
        self.max_batch_size = params['batch_size']
        self.data_file = data_file
        self.next_idx = 0
        self.feature_path = params['feature_path']
        self.params = params
        # dataset

        if mode == 'train':
            self.proposals = self.get_train_proposals(K=5)
        else:
            self.proposals = self.get_test_proposals(K=5)
            print(self.proposals)
        self.keys = list(self.proposals.keys())

        self.wv = gensim.models.KeyedVectors.load_word2vec_format(params['word2vec'],binary=True)
        # print(self.wv['dog'])
        self.ans_set = self.get_answer()
        # print(self.ans_set)
        self.gt = self.get_gt()
        self.qa_pair = self.get_qa_pair()
        self.keys = [key for key in self.keys if key in self.qa_pair and len(self.qa_pair[key]) > 0]
        print(self.keys)


        # frame / motion / question
        self.use_frame = params['use_frame']
        self.use_motion = params['use_motion']
        if self.use_frame:
            self.max_n_frames = params['max_n_frames']
            self.v_dim = params['input_video_dim']
        if self.use_motion:
            self.max_n_motions = params['max_n_motions']
            self.m_dim = params['input_motion_dim']
        self.max_q_words = params['max_n_q_words']
        self.use_qvec = params['use_qvec']
        if self.use_qvec:
            self.q_dim = params['input_ques_dim']
        # load data & initialization
        self.data_index = list(range(len(self.keys)))
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
        batch_size = min(self.max_batch_size, len(self.keys) - self.next_idx)
        if batch_size <= 0:
            return None, None, None, None, None, None, None, None

        # initialization
        if self.use_frame:
            img_frame_vecs = np.zeros((batch_size, self.max_n_frames, self.v_dim), dtype=float)
            img_frame_n = np.zeros((batch_size), dtype=int)
        else:
            img_frame_vecs = None
            img_frame_n = None
        if self.use_motion:
            img_motion_vecs = np.zeros((batch_size, self.max_n_motions, self.m_dim), dtype=float)
            img_motion_n = np.zeros((batch_size), dtype=int)
        else:
            img_motion_vecs = None
            img_motion_n = None
        if self.use_qvec:
            ques_vecs = np.zeros((batch_size, self.max_q_words, self.q_dim), dtype=float)
        else:
            ques_vecs = np.zeros((batch_size, self.max_q_words), dtype=int)
        ques_n = np.zeros((batch_size), dtype=int)
        ans_vec = np.zeros((batch_size), dtype=int)
        type_vec = np.zeros((batch_size), dtype=int)


        for i in range(batch_size):
            curr_data_index = self.data_index[self.next_idx + i]
            vid = self.keys[curr_data_index]
            if self.use_frame:
                proposal = self.proposals[vid]
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
                num_frame = self.gt[vid]['numf']
                duration = self.gt[vid]['duration']
                proposal = np.asarray(proposal)
                frames = []
                for prop in proposal:
                    feature = extract(feat,num_frame,duration,
                                      prop[0],prop[1],48,zero_padding=True)
                    if feature is None:
                        continue
                    frames.append(feature)

                if len(frames)!= len(proposal):
                    continue
                frames = np.vstack(frames)
                if len(frames) > self.max_n_frames:
                    frames = frames[:self.max_n_frames]
                n_frames = len(frames)
                img_frame_vecs[i, :n_frames, :] = frames
                img_frame_n[i] = n_frames

            # acquire question vector, answer id and type id
            tmp_ques_vecs, tmp_ans, tmp_ty = self.get_qa_vec(vid)
            if self.use_qvec:
                tmp_ques_vecs = tmp_ques_vecs[tmp_ques_vecs.sum(axis=1) != 0]  # there may be words not in pretrained models represented as all-zero vectors
                ques_vecs[i, :tmp_ques_vecs.shape[0], :] = tmp_ques_vecs
                ques_n[i] = tmp_ques_vecs.shape[0]
            else:
                ques_vecs[i, :len(tmp_ques_vecs)] = tmp_ques_vecs
                ques_n[i] = len(tmp_ques_vecs)
            ans_vec[i] = tmp_ans
            type_vec[i] = tmp_ty

        # print(ques_vecs)
        # print(ans_vec)
        self.next_idx += batch_size
        return img_frame_vecs, img_frame_n, img_motion_vecs, img_motion_n, ques_vecs, ques_n, ans_vec, type_vec

    def get_train_proposals(self,K):
        with open(self.data_file,'r') as fr:
            data = json.load(fr)

        train_proposals = dict()
        for key,item in data.items():
            proposals = item['timestamps']
            if len(proposals) <= K:
                train_proposals[key[2:]] = proposals
            else:
                train_proposals[key[2:]] = self.filter(proposals,K)
        fr.close()
        return train_proposals

    def get_test_proposals(self,K):
        with open(self.data_file, 'rb') as fr:
            data = pickle.load(fr, encoding='latin-1')

        val_proposals = dict()
        for key,item in data.items():
            val_proposals[key] = self.get_top_proposals(item,K)
        fr.close()
        return val_proposals

    def get_top_proposals(self,proposals, K):
        good_proposals = list()
        for pros in proposals:
            ranked_pros = sorted(pros, key=lambda x: x[2], reverse=False)
            good_proposals.append(ranked_pros[0])
        ranked_good_proposals = sorted(good_proposals, key=lambda x: x[2], reverse=False)

        top_proposals = list()
        top_proposals.append(ranked_good_proposals[0])
        num = len(ranked_good_proposals)
        for i in range(num):
            flag = 1
            for j in range(len(top_proposals)):
                iou_score = self.iou(ranked_good_proposals[i], top_proposals[j])
                if iou_score > 0.5:
                    flag = 0
                    break
            if flag:
                top_proposals.append(ranked_good_proposals[i])
            if len(top_proposals) >= K:
                break
        seq_top_proposals = sorted(top_proposals, key=lambda x: x[1], reverse=False)
        return seq_top_proposals

    def filter(self,proposals,K):
        proposal_mostK = list()
        dur = [proposal[1] - proposal[0] for proposal in proposals]
        score_list = [(i,dur[i]) for i in range(len(dur))]

        score_list.sort(key = lambda x: x[1],reverse=True)
        proposal_mostK.append(proposals[score_list[0][0]])
        count = 1
        for i in range(len(score_list)):
            ind = score_list[i][0]
            test_pro = proposals[ind]
            flag = True
            for target_pro in proposal_mostK:
                need = self.iou(test_pro,target_pro)
                if need >= 0.5:
                    flag = False
                    break
            if flag:
                proposal_mostK.append(test_pro)
                count += 1
                if count >= K:
                    break
        return proposal_mostK

    def iou(self,test, target):

        tt1 = np.maximum(target[0], test[0])
        tt2 = np.minimum(target[1], test[1])

        # Non-negative overlap score
        intersection = (tt2 - tt1).clip(0)
        union = ((test[1] - test[0]) +
                 (target[1] - target[0]) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        iou = intersection / union
        return iou

    def get_answer(self):
        with open(self.params['answer'],'r') as fr:
            data = json.load(fr)
        ans_set = list()
        for key,item in data.items():
            ans_set.extend(item)
        return ans_set


    def get_gt(self):
        with open(self.params['database'], 'rb') as f:
            gt = pickle.load(f)['database']
        return gt

    def get_qa_pair(self):
        with open(self.params['question'],'r') as f:
            data = json.load(f)
        qa_pair = dict()
        for key,item in data.items():
            qa_pair[key[2:]] = item
        return  qa_pair

    def get_qa_vec(self,vid):
        data = self.qa_pair[vid]
        if len(data) == 0:
            return None,None,None
        num = len(data)
        ind = random.randint(0,num-1)
        ty = data[ind][0]
        ques = data[ind][1]
        ans = data[ind][2]

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






