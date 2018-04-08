import sys
sys.path.append('./qa_utils/')
import tensorflow as tf
import layers

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
        self.lstm_dim = model_params['lstm_dim']
        self.attention_dim = model_params['attention_dim']
        #self.n_steps = model_params['lstm_step']
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

        # video LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
        input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)
        v_lstm_output, _ = layers.dynamic_origin_lstm_layer(input_x, self.lstm_dim, 'v_lstm', input_len = self.input_x_len)
        v_lstm_output = tf.contrib.layers.dropout(v_lstm_output, self.dropout_prob, is_training=self.is_training)

        # Question LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
        input_q = tf.contrib.layers.dropout(self.input_q, self.dropout_prob, is_training=self.is_training)
        q_lstm_output, q_lstm_state = layers.dynamic_origin_lstm_layer(input_q, self.lstm_dim, 'q_lstm', input_len=self.input_q_len)
        q_lstm_output = tf.contrib.layers.dropout(q_lstm_output, self.dropout_prob, is_training=self.is_training)
        q_last_state = tf.contrib.layers.dropout(q_lstm_state[1], self.dropout_prob, is_training=self.is_training)

        # local attention layer (batch_size, max_q_n_words, q_dim) , [n_steps * (batch_size, 2*lstm_dim)] -> [batch_size, 2*lstm_dim]
        v_first_attention_output, first_attention_score_list = layers.collective_matrix_attention_layer(v_lstm_output, q_lstm_output, self.attention_dim, 'v_first_local_attention', context_len=self.input_q_len, use_maxpooling=False)
        v_global_attention_output, first_attention_score = layers.matrix_attention_layer(v_lstm_output, q_last_state, self.attention_dim, 'v_global_attention')

        # video attention lstm
        v_input_att = tf.contrib.layers.dropout(v_first_attention_output, self.dropout_prob, is_training=self.is_training)
        v_att_lstm_output, _ = layers.dynamic_origin_lstm_layer(v_input_att, self.lstm_dim, 'v_att_lstm', input_len=self.input_q_len)
        v_att_lstm_output = tf.contrib.layers.dropout(v_att_lstm_output, self.dropout_prob, is_training=self.is_training)

        #att_last_state = tf.contrib.layers.dropout(att_lstm_state[1], self.dropout_prob, is_training=self.is_training)

        # second attention (batch_size, input_video_dim)
        v_second_attention_output, second_attention_score = layers.matrix_attention_layer(v_att_lstm_output, q_last_state, self.attention_dim, 'v_second_local_attention')
        print (v_second_attention_output.shape)

        self.attention = tf.reduce_sum(tf.multiply(first_attention_score_list, tf.expand_dims(second_attention_score, 2)),1)

        # dot product
        #qv_dot = tf.multiply(q_last_state, v_last_state)

        # concatenation
        concat_output = tf.concat([q_last_state, v_global_attention_output, v_second_attention_output], axis=1)

        # softmax projection [batch_size, 2*lstm_dim] -> [batch_size, n_classes]
        self.output = layers.linear_layer(concat_output, self.n_classes, 'linear')

        # sparse softmax for one-class label with regularization
        variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in variables ])
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.y, self.output) + self.regularization_beta * regularization_cost
        tf.summary.scalar('cross entropy', self.loss)

    def build_pred_proc(self):
        # return prediction status for evaluation
        # get a vector describing whether each label is in top-1 predicion
        self.predict_status = tf.nn.in_top_k(tf.nn.softmax(self.output), self.y, k=1)
        self.predict_status = tf.cast(self.predict_status, tf.int32)
        self.predict_status3 = tf.nn.in_top_k(tf.nn.softmax(self.output), self.y, k=3)
        self.predict_status3 = tf.cast(self.predict_status3, tf.int32)
        self.predict_status5 = tf.nn.in_top_k(tf.nn.softmax(self.output), self.y, k=5)
        self.predict_status5 = tf.cast(self.predict_status5, tf.int32)


    def build_model(self):
        # build all graphes
        self.build_train_proc()
        self.build_pred_proc()
