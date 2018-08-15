import sys
sys.path.append("..")
import tensorflow as tf
import qa_utils.layers as layers


class Model(object):
    def __init__(self, data_params, model_params):
        self.batch_size = data_params['batch_size']
        self.n_classes = data_params['n_classes']

        self.input_frame_dim = data_params['input_video_dim']
        self.input_n_frames = data_params['max_n_frames']
        self.input_ques_dim = data_params['input_ques_dim']
        self.max_n_q_words = data_params['max_n_q_words']
        self.ref_dim = data_params['ref_dim']
        self.lstm_dim = model_params['lstm_dim']
        self.second_lstm_dim = model_params['second_lstm_dim']
        self.attention_dim = model_params['attention_dim']
        self.regularization_beta = model_params['regularization_beta']
        self.dropout_prob = model_params['dropout_prob']

    def build_train_proc(self):
        # input layer (batch_size, n_steps, input_dim)
        self.input_q = tf.placeholder(tf.float32, [None, self.max_n_q_words, self.input_ques_dim])
        self.input_q_len = tf.placeholder(tf.int32, [None])
        self.input_x = tf.placeholder(tf.float32, [None, self.input_n_frames, self.input_frame_dim])
        self.input_x_len = tf.placeholder(tf.int32, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.y = tf.placeholder(tf.int32, [None])
        self.reward = tf.placeholder(tf.float32, [None])

        self.Wsi = tf.get_variable('Wsi', shape=[self.input_frame_dim, self.ref_dim], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.Wsh = tf.get_variable('Wsh', shape=[self.lstm_dim, self.ref_dim], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.Wsq = tf.get_variable('Wsq', shape=[self.lstm_dim, self.ref_dim], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.bias = tf.get_variable('bias', shape=[self.ref_dim], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.Vs = tf.get_variable('Vs', shape=[self.ref_dim, 1], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())

        self.input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)

        # Question LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
        self.input_q = tf.contrib.layers.dropout(self.input_q, self.dropout_prob, is_training=self.is_training)
        q_lstm_output, q_lstm_state = layers.dynamic_origin_lstm_layer(self.input_q, self.lstm_dim, 'q_lstm',
                                                                       input_len=self.input_q_len)
        q_last_state = tf.contrib.layers.dropout(q_lstm_state[1], self.dropout_prob, is_training=self.is_training)

        cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_dim, forget_bias=0.0, state_is_tuple=True)
        state = cell.zero_state(self.batch_size, tf.float32)

        cell_second = tf.contrib.rnn.BasicLSTMCell(self.second_lstm_dim, forget_bias=0.0, state_is_tuple=True)
        state_second = cell_second.zero_state(self.batch_size, tf.float32)
        # cell_output_second = tf.zeros([self.batch_size,self.lstm_dim])

        img_first_outputs = []
        img_second_outputs = []
        mask_output = []
        self.cond_list = []

        with tf.variable_scope("img_first_layer"):
            for time_step in range(self.input_n_frames - 1):
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                (cell_output, state) = cell(self.input_x[:, time_step, :], state)

                ref = tf.matmul(state[1], self.Wsh) + tf.matmul(self.input_x[:, time_step + 1, :], self.Wsi) + self.bias
                condition = tf.sigmoid(tf.matmul(ref, self.Vs))
                prod = tf.squeeze(condition, 1) > 0.7
                self.cond_list.append(condition)

                state = (tf.where(prod, state[0], tf.zeros_like(state[0])),
                         tf.where(prod, state[1], tf.zeros_like(state[1])))
                img_first_outputs.append(cell_output)

                with tf.variable_scope("img_second_layer"):
                    (cell_output_second_tmp, state_second_tmp) = cell_second(cell_output, state_second)
                cell_output_second = tf.where(prod, tf.zeros_like(cell_output_second_tmp), cell_output_second_tmp)
                img_second_outputs.append(cell_output_second)
                state_second = (tf.where(prod, state_second[0], state_second_tmp[0]),
                                tf.where(prod, state_second[1], state_second_tmp[1]))
                mask_value = tf.expand_dims(tf.to_float(prod), 1)
                self.append = mask_output.append(mask_value)

        mask_output = tf.concat(mask_output, 1)

        v_first_lstm_output = tf.reshape(tf.concat(img_first_outputs, 1), [self.batch_size, -1, self.lstm_dim])
        v_second_lstm_output = tf.reshape(tf.concat(img_second_outputs, 1), [self.batch_size, -1, self.second_lstm_dim])
        v_first_lstm_output = tf.contrib.layers.dropout(v_first_lstm_output, self.dropout_prob,
                                                        is_training=self.is_training)
        # v_second_lstm_output = tf.contrib.layers.dropout(v_second_lstm_output, self.dropout_prob, is_training=self.is_training)

        v_first_attention_output, first_attention_score = layers.matrix_attention_layer(v_first_lstm_output,
                                                                                        q_last_state,
                                                                                        self.attention_dim,
                                                                                        'v_first_attention')
        v_second_attention_output, second_attention_score = layers.matrix_attention_layer(v_second_lstm_output,
                                                                                          q_last_state,
                                                                                          self.attention_dim,
                                                                                          'v_second_attention')

        concat_output = tf.concat([q_last_state, v_first_attention_output, v_second_attention_output], axis=1)
        self.output = layers.linear_layer(concat_output, self.n_classes, 'linear')

        # sparse softmax for one-class label with regularization
        variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.y,
                                                           self.output) + self.regularization_beta * regularization_cost
        self.loss_rl = tf.losses.sparse_softmax_cross_entropy(self.y, self.output,
                                                              weights=self.reward) + self.regularization_beta * regularization_cost
        tf.summary.scalar('cross entropy', self.loss)
        tf.summary.scalar('rl cross entropy', self.loss_rl)

    def build_pred_proc(self):
        # return prediction status for evaluation
        # get a vector describing whether each label is in top-1 predicion
        self.predict_status = tf.nn.in_top_k(tf.nn.softmax(self.output), self.y, k=1)
        self.predict_status = tf.cast(self.predict_status, tf.int32)
        self.top_value, self.top_index = tf.nn.top_k(tf.nn.softmax(self.output), k=1, sorted=True)

    def build_model(self):
        # build all graphes
        self.build_train_proc()
        self.build_pred_proc()
