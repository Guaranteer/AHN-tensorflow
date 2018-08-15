import sys
sys.path.append("..")
import tensorflow as tf
import qa_utils.layers as layers


class Model(object):
    def __init__(self, data_params, model_params):
        self.batch_size = data_params['batch_size']
        self.n_classes = data_params['n_classes']

        self.input_motion_dim = data_params['input_motion_dim']
        self.input_n_motions = data_params['max_n_motions']
        self.input_frame_dim = data_params['input_video_dim']
        self.input_n_frames = data_params['max_n_frames']
        self.input_ques_dim = data_params['input_ques_dim']
        self.max_n_q_words = data_params['max_n_q_words']
        self.ref_dim = data_params['ref_dim']
        self.lstm_dim = model_params['lstm_dim']
        self.attention_dim = model_params['attention_dim']
        self.regularization_beta = model_params['regularization_beta']
        self.dropout_prob = model_params['dropout_prob']

        self.build_train_proc()
        self.build_pred_proc()

    def build_train_proc(self):
        # input layer (batch_size, n_steps, input_dim)
        self.input_q = tf.placeholder(tf.float32, [None, self.max_n_q_words, self.input_ques_dim])
        self.input_q_len = tf.placeholder(tf.int32, [None])
        self.input_m = tf.placeholder(tf.float32, [None, self.input_n_motions, self.input_motion_dim])
        self.input_m_len = tf.placeholder(tf.int32, [None])
        self.input_x = tf.placeholder(tf.float32, [None, self.input_n_frames, self.input_frame_dim])
        self.input_x_len = tf.placeholder(tf.int32, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.y = tf.placeholder(tf.int32, [None])

        self.Wsi = tf.Variable(tf.truncated_normal(shape=[self.input_frame_dim, self.ref_dim], stddev=5e-2), name='Wsi')
        self.Wsh = tf.Variable(tf.truncated_normal(shape=[self.lstm_dim, self.ref_dim], stddev=5e-2), name='Wsh')
        self.Wsq = tf.Variable(tf.truncated_normal(shape=[self.lstm_dim, self.ref_dim], stddev=5e-2), name='Wsh')
        self.bias = tf.Variable(tf.truncated_normal(shape=[self.ref_dim], stddev=5e-2), name='bias')
        self.Vs = tf.Variable(tf.truncated_normal(shape=[self.ref_dim, 1], stddev=5e-2), name='Vs')

        input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)

        # v_lstm_output, _ = layers.dynamic_origin_lstm_layer(input_x, self.lstm_dim, 'v_lstm',
        #                                                     input_len=self.input_x_len)
        # v_lstm_output = tf.contrib.layers.dropout(v_lstm_output, self.dropout_prob, is_training=self.is_training)

        # Question LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
        input_q = tf.contrib.layers.dropout(self.input_q, self.dropout_prob, is_training=self.is_training)
        q_lstm_output, q_lstm_state = layers.dynamic_origin_lstm_layer(input_q, self.lstm_dim, 'q_lstm',
                                                                       input_len=self.input_q_len)
        q_lstm_output = tf.contrib.layers.dropout(q_lstm_output, self.dropout_prob, is_training=self.is_training)
        q_last_state = tf.contrib.layers.dropout(q_lstm_state[1], self.dropout_prob, is_training=self.is_training)

        cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_dim, forget_bias=0.0, state_is_tuple=True)
        self._initial_state = cell.zero_state(self.batch_size, tf.float32)
        state = self._initial_state

        cell_second = tf.contrib.rnn.BasicLSTMCell(self.lstm_dim, forget_bias=0.0, state_is_tuple=True)
        self._initial_state_second = cell_second.zero_state(self.batch_size, tf.float32)
        state_second = self._initial_state_second
        cell_output_second = tf.zeros([self.batch_size, self.lstm_dim])

        img_outputs = []
        att_outputs = []
        cond_list = []

        with tf.variable_scope("Img_RNN"):
            for time_step in range(self.input_n_frames):
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                ref = tf.matmul(state[1], self.Wsh) + tf.matmul(input_x[:, time_step, :], self.Wsi) + self.bias
                condition = tf.sigmoid(tf.matmul(ref, self.Vs))
                prod = tf.squeeze(condition, 1) > 0.3

                state = (
                tf.where(prod, state[0], tf.zeros_like(state[0])), tf.where(prod, state[1], tf.zeros_like(state[1])))
                # state = (condition * state[0], condition * state[1])
                (cell_output, state) = cell(input_x[:, time_step, :], state)

                with tf.variable_scope("img_second_layer"):
                    (cell_output_second_tmp, state_second_tmp) = cell_second(cell_output, state_second)
                cell_output_second = tf.where(prod, cell_output_second_tmp, cell_output_second)
                state_second = (tf.where(prod, state_second_tmp[0], state_second[0]),
                                tf.where(prod, state_second_tmp[1], state_second[1]))

                img_outputs.append(cell_output)
                att_outputs.append(cell_output_second)
                # att_outputs.append(tf.where(prod, tf.zeros_like(cell_output), cell_output))
                # att_outputs.append((1-condition) * input_x[:, time_step, :])
                cond_list.append(condition)

        cond_output = tf.concat(cond_list, 1)

        v_lstm_output = tf.reshape(tf.concat(img_outputs, 1), [self.batch_size, -1, self.lstm_dim])
        # v_lstm_output = tf.contrib.layers.dropout(v_lstm_output, self.dropout_prob, is_training=self.is_training)

        v_global_attention_output, first_attention_score = layers.matrix_attention_layer(v_lstm_output, q_last_state,
                                                                                         self.attention_dim,
                                                                                         'v_global_attention')

        v_lstm_prod_output = tf.reshape(tf.concat(att_outputs, 1), [self.batch_size, -1, self.lstm_dim])

        v_input_att = tf.contrib.layers.dropout(v_lstm_prod_output, self.dropout_prob, is_training=self.is_training)
        v_att_lstm_output, _ = layers.dynamic_origin_lstm_layer(v_input_att, self.lstm_dim, 'v_att_lstm',
                                                                input_len=self.input_x_len)
        v_att_lstm_output = tf.contrib.layers.dropout(v_att_lstm_output, self.dropout_prob,
                                                      is_training=self.is_training)

        v_second_attention_output, second_attention_score = layers.matrix_attention_layer(v_lstm_prod_output,
                                                                                          q_last_state,
                                                                                          self.attention_dim,
                                                                                          'v_second_local_attention')

        concat_output = tf.concat([q_last_state, v_global_attention_output, v_second_attention_output], axis=1)

        self.output = layers.linear_layer(concat_output, self.n_classes, 'linear')

        # sparse softmax for one-class label with regularization
        variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.y,
                                                           self.output) + self.regularization_beta * regularization_cost
        tf.summary.scalar('cross entropy', self.loss)

    def build_pred_proc(self):
        # return prediction status for evaluation
        # get a vector describing whether each label is in top-1 predicion
        self.predict_status = tf.nn.in_top_k(tf.nn.softmax(self.output), self.y, k=1)
        self.predict_status = tf.cast(self.predict_status, tf.int32)
        # tf.nn.top_k(tf.nn.softmax(self.output),k = 1,sorted=True)


