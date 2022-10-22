import tensorflow as tf
import ops

def encoder(opts, inputs, reuse=tf.AUTO_REUSE, is_training=False, y=None):
    with tf.variable_scope("encoder", reuse=reuse):
        return dcgan_encoder(opts, inputs, is_training, reuse, y)


def classifier(opts, noise, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("classifier", reuse=reuse):
        if opts['mlp_classifier']:
            out = ops.linear(opts, noise, 500, 'mlp1')
            out = tf.nn.relu(out)
            out = ops.linear(opts, out, 500, 'mlp2')
            out = tf.nn.relu(out)
            logits = ops.linear(opts, out, opts['n_classes'], 'classifier')
        else:
            logits = ops.linear(opts, noise, opts['n_classes'], 'classifier')
    return logits


def decoder(opts, noise, reuse=tf.AUTO_REUSE, is_training=True):
    with tf.variable_scope("generator", reuse=reuse):
        res = dcgan_decoder(opts, noise, is_training, reuse)
        return res


def dcgan_encoder(opts, inputs, is_training=False, reuse=False, y=None):
    num_units = opts['e_num_filters']
    num_layers = opts['e_num_layers']
    layer_x = inputs
    for i in range(num_layers):
        scale = 2 ** (num_layers - i - 1)
        layer_x = ops.conv3d(opts, layer_x, num_units / scale, conv_filters_dim=[3,3,7-2*i],padding='VALID',scope='h%d_conv' % i)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x, is_training,
                                     reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    else:
        layer_x_shape = layer_x.get_shape().as_list()
        layer_x_2d = tf.reshape(layer_x,(-1,layer_x_shape[1],layer_x_shape[2],layer_x_shape[3]*layer_x_shape[4]))
        layer_x_2d = ops.conv2d(opts, layer_x_2d, output_dim = 128, padding='VALID',scope='2d_conv')
        if opts['batch_norm']:
            layer_x_2d = ops.batch_norm(opts, layer_x_2d, is_training,reuse, scope='2d_bn')
        layer_x = tf.nn.relu(layer_x_2d)

        layer_x_2d_shape = layer_x.get_shape().as_list()
        h_dim = layer_x_2d_shape[1]*layer_x_2d_shape[2]*layer_x_2d_shape[3]
        h = tf.reshape(layer_x, [-1, h_dim])
        y_onehot = tf.one_hot(y, opts['n_classes'])
        h_y = tf.concat((h, y_onehot), axis=1)
        h = ops.linear(opts, h_y, h_dim, scope='h_y_lin')
        mean = ops.linear(opts, h, opts['zdim'], scope='mean_lin')
        log_sigmas = ops.linear(opts, h,opts['zdim'], scope='log_sigmas_lin')
        return mean, log_sigmas #(?,64),(?,64)

def dcgan_decoder(opts, noise, is_training=False, reuse=False):
    output_shape = [opts['window_size'],opts['window_size'],opts['num_pcs'],1]
    num_units = opts['g_num_filters']
    batch_size = tf.shape(noise)[0]
    num_layers = opts['g_num_layers']

    height = output_shape[0] // 2 ** (num_layers - 1)
    width = output_shape[1] // 2 ** (num_layers - 1)
    depth = output_shape[2] // 2 ** (num_layers - 1)

    h0 = ops.linear(
        opts, noise,  height * width * depth * num_units, scope='h0_lin')
    h0 = tf.reshape(h0, [-1, height, width, depth, num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2 ** (i + 1)
        _out_shape = [batch_size, height * scale,
                      width * scale, depth*scale, num_units // scale]
        layer_x = ops.deconv3d(opts, layer_x, _out_shape,
                               scope='h%d_deconv' % i)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x,
                                     is_training, reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)

    last_h = ops.deconv3d(
        opts, layer_x, _out_shape, d_h=1, d_w=1, d_d =1,scope='hfinal_deconv', conv_filters_dim=[2,2,3], padding='VALID')#13*13
    return tf.nn.sigmoid(last_h)
