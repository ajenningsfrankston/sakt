# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(value=position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(params=lookup_table, ids=position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs


def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(x=inputs, axes=[-1], keepdims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma.numpy() * normalized + beta.numpy()

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        lookup_table = tf.compat.v1.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.keras.regularizers.l2(0.5 * (l2_reg)))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(params=lookup_table, ids=inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t: return outputs,lookup_table
    else: return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        sizeof_V=100,
                        with_qk=False):

    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.compat.v1.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.compat.v1.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.compat.v1.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(a=K_, perm=[0, 2, 1])) # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5) 

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(input_tensor=tf.abs(keys), axis=-1)) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(input=queries)[1], 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.compat.v1.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(input=outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.compat.v1.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        #QK=outputs
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
        QK=outputs
        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(input_tensor=tf.abs(queries), axis=-1)) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(input=keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(value=is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)

    return outputs,QK


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(value=is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(value=is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs