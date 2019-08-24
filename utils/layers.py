import tensorflow as tf
import utils.operations as op


def loss(x, y, num_classes=2, is_clip=True, clip_value=10):
    
    assert isinstance(num_classes, int)
    assert num_classes >= 2

    W = tf.get_variable(
        name='weights',
        shape=[x.shape[-1], num_classes-1],
        initializer=tf.orthogonal_initializer())
    bias = tf.get_variable(
        name='bias',
        shape=[num_classes-1],
        initializer=tf.zeros_initializer())

    logits = tf.reshape(tf.matmul(x, W) + bias, [-1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(y, tf.float32),
        logits=logits)
    if is_clip == True:
        loss = tf.reduce_mean(tf.clip_by_value(loss, -clip_value, clip_value))
    else:
        loss = tf.reduce_mean(loss)

    return loss, logits

    
def Word_Sim(Q, K, attention_type='dot', is_mask=True, mask_value=-2**32+1, drop_prob=None):
    '''calculate word-by-word similarity  u*r '''
    assert attention_type in ('dot', 'bilinear')
    if attention_type == 'dot':
        assert Q.shape[-1] == K.shape[-1]

    if attention_type == 'dot':
        logits = op.dot_sim(Q, K)
    if attention_type == 'bilinear':
        logits = op.bilinear_sim(Q, K)
    return logits


def Segment_Sim(Q, K, A_matrix):
    '''calculate segment-by-segment similarity  u*A*r'''
    assert Q.shape == K.shape and Q.shape[-1]==A_matrix.shape[0] and K.shape[-1]==A_matrix.shape[1]

    matrix2 = tf.einsum('aij,jk->aik', Q, A_matrix)
    matrix2 = op.dot_sim(matrix2, K)
    return matrix2

def consine_distance(q,a):
    '''calculate cosine sim '''
    norm_1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1)) 
    norm_2 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))

    pooled_mul_12 = tf.reduce_sum(tf.multiply(q, a), 1) 
    score = tf.div(pooled_mul_12, tf.multiply(norm_1, norm_2)+1e-8) 
    return score


def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)


def bilstm(input, rnn_units):
    cell_fw = tf.contrib.rnn.LSTMCell(rnn_units)
    cell_bw = tf.contrib.rnn.LSTMCell(rnn_units)

    H, _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw,
          cell_bw,
          input,
          dtype=tf.float32)
    H = tf.concat([H[0], H[1]], axis=2) # [n, 2u]
    return H


def attention(Q, K, V, attention_type='dot', drop_prob=None):
    '''Add attention layer.'''
    assert attention_type in ('dot', 'bilinear')
    if attention_type == 'dot':
        assert Q.shape[-1] == K.shape[-1]

    Q_time = Q.shape[1]
    K_time = K.shape[1]

    if attention_type == 'dot':
        logits = op.dot_sim(Q, K) #[batch, Q_time, K_time]
    if attention_type == 'bilinear':
        logits = op.bilinear_sim(Q, K)
    
    attention = tf.nn.softmax(logits)

    if drop_prob is not None:
        attention = tf.nn.dropout(attention, drop_prob)

    return op.weighted_sum(attention, V)


def conv_op(inputs, shape, conv_type, dilation, name):
    assert conv_type in ['conv2d','atrous_conv2d']
    W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.contrib.layers.xavier_initializer(-0.01, 0.01))
    b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
    if conv_type is 'conv2d':
        return tf.add(tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='SAME'), b)
    else:
        conv = tf.nn.atrous_conv2d(inputs, W, rate=dilation, padding='SAME')
        conv =  tf.nn.bias_add(conv, b)
        return conv


def gated_cnn(inputs, shape, conv_type, dilation, residual=True, is_layer_norm=True):
    conv_w = conv_op(inputs, shape, conv_type, dilation, "linear")
    conv_v = conv_op(inputs, shape, conv_type, dilation, "gated")
    conv = conv_w * tf.sigmoid(conv_v)

    if residual: # add shortcut on last operation
        conv = tf.add(inputs, conv)
    if is_layer_norm:
        with tf.variable_scope('residule_layer_norm'):
            conv = op.layer_norm_debug(conv)
    return conv


def agdr_block(H, repeat_times, dilation_list, filter_width, num_filters, dropout_keep_prob):
    
    
    H = tf.expand_dims(H, 1)
    with tf.variable_scope("idcnn-init-layer"):
        try:
            filter_weights = tf.get_variable(
                "cnnblock_init_filter",
                shape=[1, 1, int(H.shape[-1]), num_filters],
                initializer=tf.contrib.layers.xavier_initializer())
        except ValueError:
            tf.get_variable_scope().reuse_variables()
            filter_weights = tf.get_variable(
                "cnnblock_init_filter",
                shape=[1, 1, int(H.shape[-1]), num_filters],
                initializer=tf.contrib.layers.xavier_initializer())

        # first linear project in_channel to out_channel, the channel dim can keep same in following loop
        layerInput = tf.nn.conv2d(H, filter_weights, strides=[1, 1, 1, 1], padding="SAME", name="init_layer")
        H = tf.squeeze(layerInput, [1])
    
    StackConv = []
    for j in range(repeat_times):
        for i in range(len(dilation_list)):
            dilation = dilation_list[i]
            H = H+attention(H, H, H, attention_type='dot', drop_prob=None)

            with tf.variable_scope("atrous-conv-layer-%d-%d"%(j,i), reuse=tf.AUTO_REUSE):
                H = tf.expand_dims(H, 1)
                shape = (1, filter_width, num_filters, num_filters)
                conv = gated_cnn(H, shape, 'atrous_conv2d', dilation)
                H = tf.squeeze(conv, [1])
                #H = tf.nn.dropout(H, dropout_keep_prob)
                StackConv.append(H)

    return StackConv


def CNN_2d(x, out_channels_0, out_channels_1, dropout_keep_prob, add_relu=True, is_layer_norm=True):

    in_channels = x.shape[-1]
    weights_0 = tf.get_variable(
        name='filter_0',
        shape=[3, 3, in_channels, out_channels_0],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_0 = tf.get_variable(
        name='bias_0',
        shape=[out_channels_0],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_0 = tf.nn.conv2d(x, weights_0, strides=[1, 1, 1, 1], padding="SAME")
    conv_0 = conv_0 + bias_0

    if add_relu:
        conv_0 = tf.nn.elu(conv_0)

    pooling_0 = tf.nn.max_pool(
        conv_0, 
        ksize=[1, 3, 3, 1],
        strides=[1, 3, 3, 1], 
        padding="SAME")

    pooling_0 = tf.nn.dropout(pooling_0, dropout_keep_prob)

    #layer_1
    weights_1 = tf.get_variable(
        name='filter_1',
        shape=[3, 3, out_channels_0, out_channels_1],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_1 = tf.get_variable(
        name='bias_1',
        shape=[out_channels_1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_1 = tf.nn.conv2d(pooling_0, weights_1, strides=[1, 1, 1, 1], padding="SAME")
    conv_1 = conv_1 + bias_1

    if add_relu:
        conv_1 = tf.nn.elu(conv_1)

    pooling_1 = tf.nn.max_pool(
        conv_1, 
        ksize=[1, 3, 3, 1],
        strides=[1, 3, 3, 1], 
        padding="SAME")

    pooling_1 = tf.nn.dropout(pooling_1, dropout_keep_prob)

    return tf.contrib.layers.flatten(pooling_1)


def bigru_sequence(rnn_inputs, hidden_size, seq_lens, keep_prob):
    cell_fw = tf.nn.rnn_cell.GRUCell(hidden_size, kernel_initializer=tf.orthogonal_initializer())
    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
    
    cell_bw = tf.nn.rnn_cell.GRUCell(hidden_size, kernel_initializer=tf.orthogonal_initializer())
    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
    
    rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                               cell_bw, 
                                                               inputs=rnn_inputs, 
                                                               sequence_length=seq_lens,
                                                               dtype=tf.float32,
                                                               time_major=False
                                                               )
   
    return rnn_outputs,final_state


def intro_attention(atten_inputs, atten_size=50):
    # paper Hierarchical Attention Networks for Document Classification
    max_time = int(atten_inputs.shape[1])  
    combined_hidden_size = int(atten_inputs.shape[2])
    W_omega = tf.Variable(tf.random_normal([combined_hidden_size, atten_size], stddev=0.1, dtype=tf.float32))
    b_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))
    u_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))

    v = tf.tanh(tf.matmul(tf.reshape(atten_inputs, [-1, combined_hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    # u_omega is the summarizing question vector
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, max_time])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    atten_outs = tf.reduce_sum(atten_inputs * tf.reshape(alphas, [-1, max_time, 1]), 1)
    return atten_outs, alphas


def batch_norm(x, is_training):
    return tf.layers.batch_normalization(x, momentum=0.8, training=is_training)

