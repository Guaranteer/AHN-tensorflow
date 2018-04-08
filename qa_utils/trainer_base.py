from base_feed import Batcher
import wups
import time
import tensorflow as tf
import numpy as np
import os
import utils
import json

class Trainer(object):


    def __init__(self, train_params, model_params, data_params, model):
        self.train_params = train_params
        self.model_params = model_params
        self.data_params = data_params
        self.model = model

    def train(self):

        # training
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config = sess_config)
        # module initialization
        self.model.build_model()
        self.model_path = os.path.join(self.train_params['cache_dir'], 'tfmodel')
        self.last_checkpoint = None

        # main procedure including training & testing
        self.train_batcher = Batcher(self.data_params, self.data_params['real_train_proposals'], 'train',
                                     self.train_params['epoch_reshuffle'])
        # self.train_eval_batcher = Batcher(self.data_params, self.data_params['train_file'], 'train')
        self.valid_batcher = Batcher(self.data_params, self.data_params['real_val_proposals'], 'val')
        self.test_batcher = Batcher(self.data_params, self.data_params['real_test_proposals'], 'test')

        print ('Trainnning begins......')
        self._train(sess)
        # testing
        print ('Evaluating best model in file', self.last_checkpoint, '...')
        if self.last_checkpoint is not None:
            self.model_saver.restore(sess, self.last_checkpoint)
            self._test(sess)
        else:
            print ('ERROR: No checkpoint available!')
        sess.close()


    def _evaluate(self, sess, model, batcher):
        # evaluate the model in a set
        batcher.reset()
        type_count = np.zeros(self.data_params['n_types'], dtype=float)
        correct_count = np.zeros(self.data_params['n_types'], dtype=float)
        wups_count = np.zeros(self.data_params['n_types'], dtype=float)
        wups_count2 = np.zeros(self.data_params['n_types'], dtype=float)
        conds_list = list()


        num_per_epoch = len(batcher.data_index) // 100 + 1
        for _ in range(num_per_epoch):
            img_frame_vecs, img_frame_n, ques_vecs, ques_n, ans_vec, type_vec = batcher.generate()

            batch_data = {
                model.input_q: ques_vecs,
                model.y: ans_vec
            }

            batch_data[model.input_x] = img_frame_vecs
            batch_data[model.input_x_len] = img_frame_n
            batch_data[model.input_q_len] = ques_n
            batch_data[model.is_training] = False

            predict_status,top_index, conds = \
                sess.run([model.predict_status,model.top_index,model.cond_list], feed_dict = batch_data)

            print(top_index)
            for i in range(len(type_vec)):
                type_count[type_vec[i]] += 1
                correct_count[type_vec[i]] += predict_status[i]

                ground_a = batcher.ans_set[ans_vec[i]]
                generate_a = batcher.ans_set[top_index[i][0]]

                wups_value = wups.compute_wups([ground_a], [generate_a], 0.7)
                wups_value2 = wups.compute_wups([ground_a], [generate_a], 0.9)
                wups_count[type_vec[i]] += wups_value
                wups_count2[type_vec[i]] += wups_value2


        print(conds)
        print('****************************')
        acc = correct_count.sum() / type_count.sum()
        wup_acc = wups_count.sum() / type_count.sum()
        wup_acc2 = wups_count2.sum() / type_count.sum()
        print ('Overall Accuracy (top 1):', acc, '[', correct_count.sum(), '/', type_count.sum(), ']')
        print('Overall Wup (@0):', wup_acc, '[', wups_count.sum(), '/', type_count.sum(), ']')
        print('Overall Wup (@0.9):', wup_acc2, '[', wups_count2.sum(), '/', type_count.sum(), ']')
        type_acc = [correct_count[i] / type_count[i] for i in range(self.data_params['n_types'])]
        type_wup_acc = [wups_count[i] / type_count[i] for i in range(self.data_params['n_types'])]
        type_wup_acc2 = [wups_count2[i] / type_count[i] for i in range(self.data_params['n_types'])]
        print ('Accuracy for each type:', type_acc)
        print('Wup@0 for each type:', type_wup_acc)
        print('Wup@0.9 for each type:', type_wup_acc2)
        print(type_count)

        return acc

    def _train(self, sess):
        # tensorflow initialization
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        tf.summary.scalar('global_step',global_step)
        learning_rates = tf.train.exponential_decay(self.train_params['learning_rate'], global_step, decay_steps=self.train_params['lr_decay_n_iters'], decay_rate=self.train_params['lr_decay_rate'], staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rates)
        #optimizer = tf.train.AdamOptimizer(self.train_params['learning_rate'])
        train_proc = optimizer.minimize(self.model.loss, global_step=global_step)
        train_proc_rl = optimizer.minimize(self.model.loss_rl, global_step=global_step)

        #train_proc = optimizer.minimize(model.loss)

        self.summary_proc = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.train_params['summary_dir'], sess.graph)

        self.model_saver = tf.train.Saver()

        # training
        with open('./model_list.json','r') as fj:
            saved_model = json.load(fj)['best_model']

        if saved_model == "":
            init_proc = tf.global_variables_initializer()
            sess.run(init_proc)
        else:
            print('restore the mest model')
            self.model_saver.restore(sess, saved_model)

        best_epoch_acc = 0
        best_epoch_id = 0

        print ('****************************')
        print ('Trainning datetime:', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print ('Trainning params')
        print (self.train_params)
        print ('Model params')
        print (self.model_params)
        print ('Data params')
        print (self.data_params)
        utils.count_total_variables()
        print ('****************************')
        for i_epoch in range(self.train_params['max_epoches']):
            # train an epoch
            t_begin = time.time()
            t1 = time.time()
            self.train_batcher.reset()
            i_batch = 0
            loss_sum = 0
            num_per_epoch = len(self.train_batcher.data_index) // 100 + 1
            print(num_per_epoch)
            for _ in range(num_per_epoch):
                img_frame_vecs, img_frame_n, ques_vecs, ques_n, ans_vec, type_vec = self.train_batcher.generate()
                # train a batch
                # print(self.train_batcher.next_idx)
                if ans_vec is None:
                    break
                batch_data = {
                    self.model.input_q: ques_vecs,
                    self.model.y: ans_vec
                }

                batch_data[self.model.input_x] = img_frame_vecs
                batch_data[self.model.input_x_len] = img_frame_n
                batch_data[self.model.input_q_len] = ques_n
                batch_data[self.model.is_training] = True

                top_index = sess.run(self.model.top_index,feed_dict=batch_data)
                reward = np.zeros(len(ans_vec),dtype=float)
                for i in range(len(ans_vec)):
                    ground_a = self.train_batcher.ans_set[ans_vec[i]]
                    generate_a = self.train_batcher.ans_set[top_index[i][0]]
                    wups_value = wups.compute_wups([ground_a], [generate_a], 0)
                    reward[i] = wups_value
                batch_data[self.model.reward] = reward
                _, loss, summary, g_step, learning_r = sess.run([train_proc_rl, self.model.loss_rl, self.summary_proc, global_step, learning_rates],
                                                    feed_dict=batch_data)

                # self.summary_writer.add_summary(summary,global_step=g_step)
                # display batch info
                i_batch += 1
                loss_sum += loss
                if i_batch % self.train_params['display_batch_interval'] == 0:
                    t2 = time.time()
                    print ('Epoch %d, Batch %d, loss = %.4f, %.3f seconds/batch, learning_rates = %.4f' % (i_epoch, i_batch, loss, (t2-t1)/self.train_params['display_batch_interval'], learning_r))
                    t1 = t2

            # do summaries and evaluations
            #if i_epoch % train_params['summary_interval'] == 0:
            #    summary_str = sess.run(summary_proc, feed_dict=batch_data)
            #    summary_writer.add_summary(summary_str, i_batch)

            # print info and do early stopping
            avg_batch_loss = loss_sum/i_batch
            t_end = time.time()
            if i_epoch % self.train_params['evaluate_interval'] == 0:
                print ('****************************')
                print ('Overall evaluation')
                print ('****************************')
                _, valid_acc, _ = self._test(sess)
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                print ('****************************')
            else:
                print ('****************************')
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                valid_acc = self._evaluate(sess, self.model, self.valid_batcher)
                print ('****************************')

            if valid_acc > best_epoch_acc:
                best_epoch_acc = valid_acc
                best_epoch_id = i_epoch
                print ('Saving new best model...')
                timestamp = time.strftime("%m%d%H%M%S", time.localtime())
                self.last_checkpoint = self.model_saver.save(sess, self.model_path+timestamp, global_step=global_step)
                print ('Saved at', self.last_checkpoint)
            else:
                if i_epoch-best_epoch_id >= self.train_params['early_stopping']:
                    print ('Early stopped. Best loss %.3f at epoch %d' % (best_epoch_acc, best_epoch_id))
                    break


    def _test(self, sess):
        print ('Train set:')
        train_acc = self._evaluate(sess, self.model, self.train_batcher)
        print ('Validation set:')
        valid_acc = self._evaluate(sess, self.model, self.valid_batcher)
        print ('Test set:')
        test_acc = self._evaluate(sess, self.model, self.test_batcher)
        return train_acc, valid_acc, test_acc
