# dual layer

# # video LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
# input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)
# v_lstm_output, _ = layers.dynamic_origin_lstm_layer(input_x, self.lstm_dim, 'v_lstm',
#                                                     input_len=self.input_x_len)
# v_lstm_output = tf.contrib.layers.dropout(v_lstm_output, self.dropout_prob, is_training=self.is_training)
#
# # Question LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
# input_q = tf.contrib.layers.dropout(self.input_q, self.dropout_prob, is_training=self.is_training)
# q_lstm_output, q_lstm_state = layers.dynamic_origin_lstm_layer(input_q, self.lstm_dim, 'q_lstm',
#                                                                input_len=self.input_q_len)
# q_lstm_output = tf.contrib.layers.dropout(q_lstm_output, self.dropout_prob, is_training=self.is_training)
# q_last_state = tf.contrib.layers.dropout(q_lstm_state[1], self.dropout_prob, is_training=self.is_training)
#
# # local attention layer (batch_size, max_q_n_words, q_dim) , [n_steps * (batch_size, 2*lstm_dim)] -> [batch_size, 2*lstm_dim]
# v_first_attention_output, first_attention_score_list = layers.collective_matrix_attention_layer(v_lstm_output,
#                                                                                                 q_lstm_output,
#                                                                                                 self.attention_dim,
#                                                                                                 'v_first_local_attention',
#                                                                                                 context_len=self.input_q_len,
#                                                                                                 use_maxpooling=False)
# v_global_attention_output, first_attention_score = layers.matrix_attention_layer(v_lstm_output, q_last_state,
#                                                                                  self.attention_dim,
#                                                                                  'v_global_attention')
#
# # video attention lstm
# v_input_att = tf.contrib.layers.dropout(v_first_attention_output, self.dropout_prob,
#                                         is_training=self.is_training)
# v_att_lstm_output, _ = layers.dynamic_origin_lstm_layer(v_input_att, self.lstm_dim, 'v_att_lstm',
#                                                         input_len=self.input_q_len)
# v_att_lstm_output = tf.contrib.layers.dropout(v_att_lstm_output, self.dropout_prob,
#                                               is_training=self.is_training)
#
# # att_last_state = tf.contrib.layers.dropout(att_lstm_state[1], self.dropout_prob, is_training=self.is_training)
#
# # second attention (batch_size, input_video_dim)
# v_second_attention_output, second_attention_score = layers.matrix_attention_layer(v_att_lstm_output,
#                                                                                   q_last_state,
#                                                                                   self.attention_dim,
#                                                                                   'v_second_local_attention')
# print(v_second_attention_output.shape)
#
# self.attention = tf.reduce_sum(
#     tf.multiply(first_attention_score_list, tf.expand_dims(second_attention_score, 2)), 1)
#
# # dot product
# # qv_dot = tf.multiply(q_last_state, v_last_state)
#
# # concatenation
# concat_output = tf.concat([q_last_state, v_global_attention_output, v_second_attention_output], axis=1)
#
# # softmax projection [batch_size, 2*lstm_dim] -> [batch_size, n_classes]
# self.output = layers.linear_layer(concat_output, self.n_classes, 'linear')

################
##  VQA model ##
################
# lstm_dim = 1024
# # video LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
# input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)
# _, v_lstm_state = layers.dynamic_origin_lstm_layer(input_x, lstm_dim, 'v_lstm', input_len = self.input_x_len)

# # question LSTM layer
# q_lstm_output, q_lstm_state1 = layers.dynamic_origin_lstm_layer(self.input_q, 512, 'q_lstm', input_len = self.input_q_len)
# _, q_lstm_state2 = layers.dynamic_origin_lstm_layer(q_lstm_output, 512, 'q_lstm1', input_len = self.input_q_len)

# q_lstm_state_temp = tf.concat([q_lstm_state1[1], q_lstm_state2[1]], 1)
# q_lstm_state = layers.linear_layer(q_lstm_state_temp, 1024, 'linear0')

# qv_dot = tf.multiply(q_lstm_state, v_lstm_state[1])    # [None, 1024]

# # softmax projection [batch_size, 2*lstm_dim] -> [batch_size, n_classes]
# self.output = layers.linear_layer(qv_dot, self.n_classes, 'linear')



################
##  SS model  ##
################
# lstm_dim = 1024
#
# # decode video
# # video LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
# input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)
# _, v_lstm_state = layers.dynamic_origin_lstm_layer(input_x, lstm_dim, 'v_lstm', input_len = self.input_x_len)
# v_last_state = tf.contrib.layers.dropout(v_lstm_state, self.dropout_prob, is_training=self.is_training)
# # decode question
# scope_name = '1_lstm'
# with tf.variable_scope(scope_name):
#     lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim)
#     lstm_output, final_state = tf.nn.dynamic_rnn(lstm_cell, self.input_q, initial_state=v_lstm_state, sequence_length=self.input_q_len, dtype=tf.float32, scope=scope_name)
# q_last_state = tf.contrib.layers.dropout(final_state[1], self.dropout_prob, is_training=self.is_training)
#
# self.output = layers.linear_layer(q_last_state, self.n_classes, 'linear')




################
##  MN model  ##
################

# # bi-decode video
# lstm_dim = 512
# input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)
# v_lstm_out, v_lstm_state = layers.dynamic_origin_bilstm_layer(input_x, lstm_dim, 'v_lstm_bi', input_len = self.input_x_len)

# # decode question
# _, q_lstm_state = layers.dynamic_origin_lstm_layer(self.input_q, lstm_dim, '1_lstm', input_len = self.input_q_len)
# q_last_state = tf.contrib.layers.dropout(q_lstm_state[1], self.dropout_prob, is_training=self.is_training)

# v_first_attention_output, _ = layers.matrix_attention_layer(v_lstm_out, q_last_state, self.attention_dim, 'v_first_local_attention')

# concat_output = tf.concat([v_first_attention_output, q_last_state], axis=1)

# # softmax projection [batch_size, 2*lstm_dim] -> [batch_size, n_classes]
# self.output = layers.linear_layer(concat_output, self.n_classes, 'linear')


################
##  Dual BASE ##
################
# hidden_dim_1 = 512
# K_layer_1 = 2

# input_m = tf.contrib.layers.dropout(self.input_m, self.dropout_prob, is_training=self.is_training)
# input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)
# input_q = tf.contrib.layers.dropout(self.input_q, self.dropout_prob, is_training=self.is_training)

# Hp_0, _ = layers.dynamic_origin_bilstm_layer(input_x, hidden_dim_1, 'BiFrame_qa', input_len=self.input_x_len)   # (?, input_x_len, 2*hidden_dim) (?, 60, 512)
# Hm_0, _ = layers.dynamic_origin_bilstm_layer(input_m, hidden_dim_1, 'BiFrame_qa2', input_len=self.input_m_len)   # (?, input_x_len, 2*hidden_dim) (?, 60, 512)
# Hq, Hq_last_state = layers.dynamic_origin_bilstm_layer(input_q, hidden_dim_1, 'BiQuestion_qa', input_len=self.input_q_len)  # (?, input_q_len, 2*hidden_dim)

# Hp_output = layers.gated_attention_layer(Hp_0, Hq, hidden_dim_1, K_layer_1, "gate_att_qa_x")
# Hm_output = layers.gated_attention_layer(Hm_0, Hq, hidden_dim_1, K_layer_1, "gate_att_qa_m")

# Hq_last_state = tf.concat(Hq_last_state, 0)
# Hq_last_state = tf.contrib.layers.dropout(Hq_last_state[1], self.dropout_prob, is_training=self.is_training)

# # Question LSTM layer, [batch_size, max_n_q_words, input_ques_dim] -> [batch_size, max_n_q_words, lstm_dim]
# # q_last_state [batch_size, lstm_dim]
# q_lstm_output, q_lstm_state = layers.dynamic_origin_lstm_layer(input_q, self.lstm_dim, 'q_lstm', input_len=self.input_q_len)
# q_lstm_output = tf.contrib.layers.dropout(q_lstm_output, self.dropout_prob, is_training=self.is_training)
# q_last_state = tf.contrib.layers.dropout(q_lstm_state[1], self.dropout_prob, is_training=self.is_training)

# m_f_attention_output, _ = layers.matrix_attention_layer(tf.stack([Hp_output, Hm_output], axis=1), q_last_state, self.attention_dim, 'm_f_attention_output')

# # concatenation
# concat_output = tf.concat([Hp_output, Hm_output, Hq_last_state, m_f_attention_output], axis=1)          # [2221 * attention_dim]
# self.output = layers.linear_layer(concat_output, self.n_classes, 'linear')