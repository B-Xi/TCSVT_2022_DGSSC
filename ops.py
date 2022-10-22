import numpy as np
import tensorflow as tf

def batch_norm(opts, _input, is_train, reuse, scope, scale=True):
    """Batch normalization based on tf.contrib.layers.
    """
    return tf.contrib.layers.batch_norm(
        _input, center=True, scale=scale,
        epsilon=opts['batch_norm_eps'], decay=opts['batch_norm_decay'],
        is_training=is_train, reuse=reuse, updates_collections=None,
        scope=scope, fused=False)


def linear(opts, input_, output_dim, scope=None, init='normal', reuse=None):
    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()

    assert len(shape) > 0
    in_shape = shape[1]
    if len(shape) > 2:
        input_ = tf.reshape(input_, [-1, np.prod(shape[1:])])
        in_shape = np.prod(shape[1:])

    with tf.variable_scope(scope or "lin", reuse=reuse):
        if init == 'normal':
            matrix = tf.get_variable(
                "W", [in_shape, output_dim], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        else:
            matrix = tf.get_variable(
                "W", [in_shape, output_dim], tf.float32,
                tf.constant_initializer(np.identity(in_shape)))
        bias = tf.get_variable(
            "b", [output_dim],
            initializer=tf.constant_initializer(bias_start))

    return tf.matmul(input_, matrix) + bias

def conv3d(opts, input_, output_dim, d_d=1, d_h=1, d_w=1, scope=None,
           conv_filters_dim=None, padding='SAME', l2_norm=False):
    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
        k_h = conv_filters_dim
        k_w = k_h
        k_d = k_w
    else:
        k_h = conv_filters_dim[0]
        k_w = conv_filters_dim[1]
        k_d = conv_filters_dim[2]
    assert len(shape) == 5, 'Conv3d works only with 5d tensors.'
    with tf.variable_scope(scope or 'conv3d'):
        w = tf.get_variable(
            'filter', [k_h, k_w, k_d, shape[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if l2_norm:
            w = tf.nn.l2_normalize(w, 2)
        conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_d, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(bias_start))
        conv = tf.nn.bias_add(conv, biases)
    return conv


def deconv3d(opts, input_, output_shape, d_h=2, d_w=2, d_d=2, scope=None, conv_filters_dim=None, padding='SAME'):
    stddev = opts['init_std']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
        k_h = conv_filters_dim
        k_w = k_h
        k_d = k_w
    else:
        k_h = conv_filters_dim[0]
        k_w = conv_filters_dim[1]
        k_d = conv_filters_dim[2]

    assert len(shape) == 5, 'Conv3d_transpose works only with 5d tensors.'
    assert len(output_shape) == 5, 'outut_shape should be 5dimensional'

    with tf.variable_scope(scope or "deconv3d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, k_d, output_shape[-1], shape[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv3d_transpose(
            input_, w, output_shape=output_shape,
            strides=[1, d_h, d_w, d_d, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

    return deconv

def conv2d(opts, input_, output_dim, d_h=1, d_w=1, scope=None,
           conv_filters_dim=None, padding='VALID', l2_norm=False):
    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d works only with 4d tensors.'

    with tf.variable_scope(scope or 'conv2d'):
        w = tf.get_variable(
            'filter', [k_h, k_w, shape[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if l2_norm:
            w = tf.nn.l2_normalize(w, 2)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(bias_start))
        conv2d = tf.nn.bias_add(conv, biases)

    return conv2d