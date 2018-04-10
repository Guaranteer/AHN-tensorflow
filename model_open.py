import sys
sys.path.append('./qa_utils/')
import tensorflow as tf
import layers
import random
import pickle as pkl
import utils

def load_file(filename):
    with open(filename,'rb') as f1:
        return pkl.load(f1)

class Model(object):

    def __init__(self, data_params, model_params):
        #self.batch_size = data_params['batch_size']
        self.data_params = data_params
        self.model_params = model_params
        self.input_frame_dim = data_params['input_video_dim']
        self.input_n_frames = data_params['max_n_frames']
        self.input_ques_dim = data_params['input_ques_dim']
        self.max_n_q_words = data_params['max_n_q_words']
        self.max_n_a_words = data_params['max_n_a_words']
        self.n_words = data_params['n_words']

        self.ref_dim =  model_params['ref_dim']
        self.lstm_dim = model_params['lstm_dim']
        self.second_lstm_dim = model_params['second_lstm_dim']
        self.attention_dim = model_params['attention_dim']
        self.regularization_beta = model_params['regularization_beta']
        self.dropout_prob = model_params['dropout_prob']

        self.decode_dim = model_params['decode_dim']


    def build_train_proc(self):
        # input layer (batch_size, n_steps, input_dim)
        self.input_q = tf.placeholder(tf.float32, [None, self.max_n_q_words, self.input_ques_dim])
        self.input_q_len = tf.placeholder(tf.int32, [None])
        self.input_x = tf.placeholder(tf.float32, [None, self.input_n_frames, self.input_frame_dim])
        self.input_x_len = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None, self.max_n_a_words])
        self.y_mask = tf.placeholder(tf.float32, [None, self.max_n_a_words])
        self.ans_vec = tf.placeholder(tf.float32, [None, self.max_n_a_words, self.input_ques_dim])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.is_training = tf.placeholder(tf.bool)
        self.reward = tf.placeholder(tf.float32, [None])


        
        self.Wsi = tf.get_variable('Wsi', shape=[self.input_frame_dim, self.ref_dim], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        self.Wsh = tf.get_variable('Wsh', shape=[self.lstm_dim, self.ref_dim], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.Wsq = tf.get_variable('Wsq', shape=[self.lstm_dim, self.ref_dim], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.bias = tf.get_variable('bias', shape=[self.ref_dim], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.Vs = tf.get_variable('Vs', shape=[self.ref_dim,1], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        self.input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)

        # Question LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
        self.input_q = tf.contrib.layers.dropout(self.input_q, self.dropout_prob, is_training=self.is_training)
        q_lstm_output, q_lstm_state = layers.dynamic_origin_lstm_layer(self.input_q, self.lstm_dim, 'q_lstm', input_len=self.input_q_len)
        self.q_last_state = tf.contrib.layers.dropout(q_lstm_state[1], self.dropout_prob, is_training=self.is_training)


        cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_dim, forget_bias=0.0, state_is_tuple=True)
        state = cell.zero_state(self.batch_size, tf.float32)

        cell_second = tf.contrib.rnn.BasicLSTMCell(self.second_lstm_dim, forget_bias=0.0, state_is_tuple=True)
        state_second = cell_second.zero_state(self.batch_size, tf.float32)

        img_first_outputs = []
        img_second_outputs = []
        mask_output = []
        cond_list = []

        with tf.variable_scope("img_first_layer"):
            for time_step in range(self.input_n_frames-1):
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                (cell_output, state) = cell(self.input_x[:, time_step, :], state)

                ref = tf.matmul(state[1],self.Wsh) + tf.matmul(self.input_x[:, time_step+1, :],self.Wsi) + self.bias
                condition = tf.sigmoid(tf.matmul(ref,self.Vs))
                prod = tf.squeeze(condition,1) > 0.3
                cond_list.append(condition)

                state = (tf.where(prod, state[0],tf.zeros_like(state[0])),
                         tf.where(prod, state[1],tf.zeros_like(state[1])))
                img_first_outputs.append(cell_output)

                with tf.variable_scope("img_second_layer"):
                    (cell_output_second_tmp, state_second_tmp) = cell_second(cell_output, state_second)
                cell_output_second = tf.where(prod, tf.zeros_like(cell_output_second_tmp), cell_output_second_tmp)
                img_second_outputs.append(cell_output_second)
                state_second = (tf.where(prod, state_second[0], state_second_tmp[0]),
                                tf.where(prod, state_second[1], state_second_tmp[1]))

                mask_value = tf.expand_dims(tf.to_float(prod),1)
                mask_output.append(mask_value)

        mask_output = tf.concat(mask_output,1)

        self.v_first_lstm_output = tf.reshape(tf.concat(img_first_outputs, 1), [-1,self.input_n_frames-1, self.lstm_dim])
        self.v_second_lstm_output = tf.reshape(tf.concat(img_second_outputs, 1), [-1,self.input_n_frames-1, self.lstm_dim])
        self.v_first_lstm_output = tf.contrib.layers.dropout(self.v_first_lstm_output, self.dropout_prob,is_training=self.is_training)


        self.v_first_attention_output, self.first_attention_score = layers.matrix_attention_layer(self.v_first_lstm_output, self.q_last_state, self.attention_dim, 'v_first_attention')
        self.v_second_attention_output, self.second_attention_score = layers.mask_matrix_attention_layer(self.v_second_lstm_output, self.q_last_state, self.attention_dim, mask_output, 'v_second_attention')

        concat_output = tf.concat([self.q_last_state, self.v_first_attention_output], axis=1)


        # decoder

        # output -> first_atten
        # self.decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.decode_dim)
        self.decoder_cell = tf.contrib.rnn.GRUCell(self.decode_dim)

        with tf.variable_scope('linear'):
            decoder_input_W = tf.get_variable('w', shape=[concat_output.shape[1], self.decode_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.03))
            decoder_input_b = tf.get_variable('b', shape=[self.decode_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.03))
            self.decoder_input = tf.matmul(concat_output, decoder_input_W) + decoder_input_b  # [None, decode_dim]

        # answer->word predict
        self.embed_word_W = tf.Variable(tf.random_uniform([self.decode_dim, self.n_words], -0.1, 0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.random_uniform([self.n_words], -0.1, 0.1), name='embed_word_b')

        # word dim -> decode_dim
        self.word_to_lstm_w = tf.Variable(tf.random_uniform([self.input_ques_dim, self.decode_dim], -0.1, 0.1), name='word_to_lstm_W')
        self.word_to_lstm_b = tf.Variable(tf.random_uniform([self.decode_dim], -0.1, 0.1), name='word_to_lstm_b')

        # decoder attention layer
        with tf.variable_scope('decoder_attention'):
            self.attention_w_q = tf.get_variable('attention_w_q', shape=[self.lstm_dim, self.attention_dim], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            self.attention_w_x = tf.get_variable('attention_w_x', shape=[self.lstm_dim, self.attention_dim], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            self.attention_w_h = tf.get_variable('attention_w_h', shape=[self.decode_dim,self.attention_dim], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            self.attention_b = tf.get_variable('attention_b', shape=[self.attention_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            self.attention_a = tf.get_variable('attention_a', shape=[self.attention_dim, 1], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            self.attention_to_decoder = tf.get_variable('attention_to_decoder', shape=[self.lstm_dim, self.decode_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        # decoder
        with tf.variable_scope('decoder'):
            self.decoder_r = tf.get_variable('decoder_r', shape=[self.decode_dim*3, self.decode_dim], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            self.decoder_z = tf.get_variable('decoder_z', shape=[self.decode_dim*3, self.decode_dim], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            self.decoder_w = tf.get_variable('decoder_w', shape=[self.decode_dim*3,self.decode_dim], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())


        # embedding layer
        embeddings = load_file(self.data_params['word_embedding'])
        self.Wemb = tf.constant(embeddings, dtype=tf.float32)

        # generate training
        answer_train, train_loss = self.generate_answer_on_training()
        answer_test, test_loss = self.generate_answer_on_testing()

        # final
        variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in variables ])
        self.answer_word_train = answer_train
        self.train_loss = train_loss  + self.regularization_beta * regularization_cost

        self.answer_word_test = answer_test
        self.test_loss = test_loss  + self.regularization_beta * regularization_cost
        tf.summary.scalar('training cross entropy', self.train_loss)


    def generate_answer_on_training(self):
        with tf.variable_scope("decoder"):
            answer_train = []
            decoder_state = self.decoder_cell.zero_state(self.batch_size, tf.float32)
            loss = 0.0

            with tf.variable_scope("lstm") as scope:
                for i in range(self.max_n_a_words):
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        scope.reuse_variables()
                        # next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        # current_emb = tf.nn.xw_plus_b(next_word_vec, self.word_to_lstm_w, self.word_to_lstm_b)
                        current_emb = tf.nn.xw_plus_b(self.ans_vec[:, i-1, :], self.word_to_lstm_w, self.word_to_lstm_b)


                    # decoder_state
                    tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1), tf.stack([1, self.input_n_frames-1, 1]))
                    tiled_q_last_state = tf.tile(tf.expand_dims(self.q_last_state, 1),tf.stack([1, self.input_n_frames - 1, 1]))
                    attention_input = tf.tanh(utils.tensor_matmul(self.v_first_lstm_output, self.attention_w_x)
                                              + utils.tensor_matmul(tiled_q_last_state, self.attention_w_q)
                                              + utils.tensor_matmul(tiled_decoder_state_h, self.attention_w_h)
                                              +self.attention_b)
                    attention_score = tf.nn.softmax(tf.squeeze(utils.tensor_matmul(attention_input, self.attention_a), axis=[2]))
                    attention_output = tf.reduce_sum(tf.multiply(self.v_first_lstm_output, tf.expand_dims(attention_score, 2)), 1)
                    attention_decoder = tf.matmul(attention_output,self.attention_to_decoder)

                    # decoder : GRU with attention
                    decoder_input = tf.concat([decoder_state,attention_decoder,current_emb],axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_z))
                    decoder_middle = tf.concat([tf.multiply(decoder_r_t,decoder_state),tf.multiply(decoder_r_t,attention_decoder),current_emb],axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.decoder_w))
                    decoder_state = tf.multiply((1-decoder_z_t),decoder_state) + tf.multiply(decoder_z_t,decoder_state_)

                    output = decoder_state


                    # ground truth
                    labels = tf.expand_dims(self.y[:, i], 1)
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat([indices, labels], 1)
                    onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    max_prob_word = tf.argmax(logit_words, 1)
                    answer_train.append(max_prob_word)

                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
                    cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask[:,i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask[:,1:])
            return answer_train, loss

    def generate_answer_on_testing(self):
        with tf.variable_scope("decoder"):
            answer_test = []
            decoder_state = self.decoder_cell.zero_state(self.batch_size, tf.float32)
            loss = 0.0

            with tf.variable_scope("lstm") as scope:
                for i in range(self.max_n_a_words):
                    scope.reuse_variables()
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        current_emb = tf.nn.xw_plus_b(next_word_vec, self.word_to_lstm_w, self.word_to_lstm_b)

                    # decoder_state
                    tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1),
                                                    tf.stack([1, self.input_n_frames - 1, 1]))
                    tiled_q_last_state = tf.tile(tf.expand_dims(self.q_last_state, 1),
                                                 tf.stack([1, self.input_n_frames - 1, 1]))
                    attention_input = tf.tanh(utils.tensor_matmul(self.v_first_lstm_output, self.attention_w_x)
                                              + utils.tensor_matmul(tiled_q_last_state, self.attention_w_q)
                                              + utils.tensor_matmul(tiled_decoder_state_h, self.attention_w_h)
                                              + self.attention_b)
                    attention_score = tf.nn.softmax(
                        tf.squeeze(utils.tensor_matmul(attention_input, self.attention_a), axis=[2]))
                    attention_output = tf.reduce_sum(
                        tf.multiply(self.v_first_lstm_output, tf.expand_dims(attention_score, 2)), 1)
                    attention_decoder = tf.matmul(attention_output, self.attention_to_decoder)

                    # decoder : GRU with attention
                    decoder_input = tf.concat([decoder_state, attention_decoder, current_emb], axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_z))
                    decoder_middle = tf.concat(
                        [tf.multiply(decoder_r_t, decoder_state), tf.multiply(decoder_r_t, attention_decoder),
                         current_emb], axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.decoder_w))
                    decoder_state = tf.multiply((1 - decoder_z_t), decoder_state) + tf.multiply(decoder_z_t,
                                                                                                decoder_state_)

                    output = decoder_state

                    # ground truth
                    labels = tf.expand_dims(self.y[:, i], 1) 
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat([indices, labels], 1)
                    onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    max_prob_word = tf.argmax(logit_words, 1)
                    answer_test.append(max_prob_word)

                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
                    cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask[:,i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask[:,1:])
            return answer_test, loss
        

    
    def build_model(self):
        # build all graphes
        self.build_train_proc()
