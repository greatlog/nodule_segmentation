import tensorflow as tf
import pickle
import numpy as np
import logging

INPUT_SIZE = 512
NET_DEPTH = 5
batch_size = 1
keep_prob = 1
spatial_keep_prob = 1

def print_activations(t):
    print(t.op.name, " ", t.get_shape().as_list())

def gaussian_noise_layer(input_layer, std=0.05):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def filter_for_depth(depth):  # compute how many out features for each layer
    return 2 ** (6 + depth)

def Conv2DLayer(input, W_shape, b_shape):

    W = tf.Variable(initial_value=tf.truncated_normal(shape=W_shape))
    b = tf.Variable(initial_value=tf.zeros(shape=b_shape))
    output = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding="VALID")
    output = tf.nn.bias_add(output, b)
    output = leaky_relu(output, 0.01)
    print_activations(output)
    return output

def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def MaxPool2DLayer(input):
    output = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    print_activations(output)
    return output

def Conv2DLayer_s2(input, W_shape, b_shape):
    W = tf.Variable(initial_value=tf.truncated_normal(shape=W_shape))
    b = tf.Variable(initial_value=tf.zeros(shape=b_shape))
    output = tf.nn.conv2d(input, W, [1, 2, 2, 1], padding="VALID")
    output = tf.nn.bias_add(output, b)
    output = leaky_relu(output, 0.01)
    print_activations(output)
    return output

def spatial_dropout_layer(input, keep_prob, deterministic=False, rescale=True):
    """
    Parameters
    ----------
    input : tensor
        output from the previous layer
    deterministic : bool
        If true dropout and scaling is disabled, see notes
    """
    if deterministic or keep_prob == 1.:
        return input
    else:
        retain_prob = tf.constant(keep_prob, dtype=tf.float32)
        if rescale:
            input /= retain_prob

        shape = tf.shape(input)
        shape_batch = shape[0]
        shape_y = shape[1]
        shape_x = shape[2]
        shape_channel = tf.cast(shape[3], tf.float32)
        n_keep_channels = tf.floor(keep_prob * shape_channel)
        n_drop_channels = shape_channel - n_keep_channels  # must has dtype float, cannot be int
        n_keep_channels = tf.cast(n_keep_channels, dtype=tf.int32)
        n_drop_channels = tf.cast(n_drop_channels, dtype=tf.int32)
        ones = tf.ones([n_keep_channels, shape_batch, shape_y, shape_x], dtype=tf.float32)
        zeros = tf.zeros([n_drop_channels, shape_batch, shape_y, shape_x], dtype=tf.float32)
        mask = tf.concat([ones, zeros], axis=0)
        mask = tf.random_shuffle(mask)
        mask = tf.transpose(mask, [1, 2, 3, 0])

        return input * mask

def batch_norm(x, beta, gamma, mean_var_dict, depth):
    mean = np.array(mean_var_dict['mean_{}'.format(depth)])
    mean = tf.convert_to_tensor(mean,tf.float32)
    var = np.array(mean_var_dict['var_{}'.format(depth)])
    var = tf.convert_to_tensor(var,tf.float32)
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    print_activations(normed)
    return normed

def model(input_var, var_dict, mean_var_dict, keep_prob, spatial_keep_prob):
    net = {}
    net['input'] = input_var

    # if True:  # P.GAUSSIAN_NOISE > 0:
    #     net['input'] = gaussian_noise_layer(net['input'], std=0.05)

    def contraction(depth, deepest):
        n_filters = filter_for_depth(depth)
        incoming = net['input'] if depth == 0 else net['pool{}'.format(depth - 1)]

        net['conv{}_1'.format(depth)] = Conv2DLayer(incoming,
                                                    var_dict['W_conv{}_1'.format(depth)],
                                                    var_dict['b_conv{}_1'.format(depth)])
        net['conv{}_2'.format(depth)] = Conv2DLayer(net['conv{}_1'.format(depth)],
                                                    var_dict['W_conv{}_2'.format(depth)],
                                                    var_dict['b_conv{}_2'.format(depth)])

        if True:  # P.BATCH_NORMALIZATION:
            net['conv{}_2'.format(depth)] = batch_norm(net['conv{}_2'.format(depth)],
                                                       var_dict['beta_conv{}_2'.format(depth)],
                                                       var_dict['gamma_conv{}_2'.format(depth)],
                                                       mean_var_dict, depth)
        if not deepest:
            net['pool{}'.format(depth)] = MaxPool2DLayer(net['conv{}_2'.format(depth)])

    def expansion(depth, deepest):
        n_filters = filter_for_depth(depth)

        incoming = net['conv{}_2'.format(depth + 1)] if deepest else net['_conv{}_2'.format(depth + 1)]

        if depth == 3:
            upscaling = tf.image.resize_bilinear(incoming, [96, 96])
        elif depth == 2:
            upscaling = tf.image.resize_bilinear(incoming, [44 * 4, 44 * 4])
        elif depth == 1:
            upscaling = tf.image.resize_bilinear(incoming, [84 * 4, 84 * 4])
        elif depth == 0:
            upscaling = tf.image.resize_bilinear(incoming, [164 * 4, 164 * 4])

        # upscaling = Upscale2DLayer(incoming, 4)
        net['upconv{}'.format(depth)] = Conv2DLayer_s2(upscaling,
                                                       var_dict['W_upconv{}_1'.format(depth)],
                                                       var_dict['b_upconv{}_1'.format(depth)])

        print("upconv{}: ".format(depth), net['upconv{}'.format(depth)])
        if spatial_keep_prob < 1:  # P.SPATIAL_DROPOUT > 0:
            bridge_from = spatial_dropout_layer(net['conv{}_2'.format(depth)], spatial_keep_prob)
        else:
            bridge_from = net['conv{}_2'.format(depth)]
        print("bridge_from0: ", bridge_from)

        # shape = tf.shape(net['upconv{}'.format(depth)])
        shape = tf.shape(net['upconv{}'.format(depth)])
        bridge_from = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, shape[1], shape[2]), bridge_from)
        # bridge_from = tf.reshape(bridge_from, [batch_size, shape[1], shape[2], 2**(depth+6)])
        print("bridge_from: ", bridge_from)
        net['bridge{}'.format(depth)] = tf.concat([net['upconv{}'.format(depth)],
                                                   bridge_from],
                                                  axis=3)
        # net['bridge{}'.format(depth)] = batch_norm(net['bridge{}'.format(depth)],
        #                                            var_dict['beta_bridge{}'.format(depth)],
        #                                            var_dict['gamma_bridge{}'.format(depth)],
        #                                            train_phase
        #                                            )

        net['_conv{}_1'.format(depth)] = Conv2DLayer(net['bridge{}'.format(depth)],
                                                     var_dict["W__conv{}_1".format(depth)],
                                                     var_dict["b__conv{}_1".format(depth)])

        # if P.BATCH_NORMALIZATION:
        #    net['_conv{}_1'.format(depth)] = batch_norm(net['_conv{}_1'.format(depth)])

        if keep_prob < 1:  # P.DROPOUT > 0:
            net['_conv{}_1'.format(depth)] = tf.nn.dropout(net['_conv{}_1'.format(depth)], tf.constant(keep_prob))

        net['_conv{}_2'.format(depth)] = Conv2DLayer(net['_conv{}_1'.format(depth)],
                                                     var_dict['W__conv{}_2'.format(depth)],
                                                     var_dict['b__conv{}_2'.format(depth)])

    for d in range(NET_DEPTH):
        # There is no pooling at the last layer
        deepest = d == NET_DEPTH - 1
        contraction(d, deepest)

    for d in reversed(range(NET_DEPTH - 1)):
        deepest = d == NET_DEPTH - 2
        expansion(d, deepest)

    # Output layer
    net['out'] = Conv2DLayer(net['_conv0_2'], var_dict["W_out"], var_dict["b_out"])

    logging.info('Network output shape ' + str(tf.shape(net['out'])))
    return net['out']


# def infer(img, label, path_image, path_label, sess, graph):
def pre_trained_unet2d(data):

    var_dict_shape = {'W__conv0_2': (3, 3, 64, 64), 'W__conv0_1': (3, 3, 128, 64),
                      'W_conv2_2': (3, 3, 256, 256), 'beta_conv4_2': (1024,),
                      'W_conv2_1': (3, 3, 128, 256), 'b_conv3_2': (512,), 'W_conv3_2': (3, 3, 512, 512),
                      'W_conv3_1': (3, 3, 256, 512), 'b_conv3_1': (512,), 'W__conv3_1': (3, 3, 1024, 512),
                      'b_conv2_2': (256,), 'b_conv2_1': (256,), 'W__conv3_2': (3, 3, 512, 512),
                      'beta_conv2_2': (256,), 'b_out': (2,), 'b_upconv0_1': (64,), 'b_conv4_1': (1024,),
                      'W__conv2_1': (3, 3, 512, 256), 'W__conv2_2': (3, 3, 256, 256), 'b_conv4_2': (1024,),
                      'W_upconv0_1': (2, 2, 128, 64), 'b_upconv1_1': (128,), 'gamma_conv1_2': (128,),
                      'gamma_bridge2': (512,), 'gamma_bridge3': (1024,), 'gamma_bridge0': (128,),
                      'gamma_bridge1': (256,), 'beta_conv1_2': (128,), 'b_conv1_2': (128,),
                      'b_conv1_1': (128,), 'beta_conv0_2': (64,), 'W_out': (1, 1, 64, 2),
                      'b_upconv2_1': (256,), 'b_conv0_1': (64,), 'b_conv0_2': (64,), 'b_upconv3_1': (512,),
                      'beta_conv3_2': (512,), 'b__conv1_1': (128,), 'b__conv1_2': (128,), 'W_upconv1_1': (2, 2, 256, 128),
                      'W_conv4_1': (3, 3, 512, 1024), 'gamma_conv0_2': (64,), 'W_conv4_2': (3, 3, 1024, 1024),
                      'beta_bridge2': (512,), 'beta_bridge3': (1024,), 'beta_bridge0': (128,), 'beta_bridge1': (256,),
                      'gamma_conv2_2': (256,), 'W_upconv3_1': (2, 2, 1024, 512), 'W_conv0_1': (3, 3, 1, 64),
                      'W_conv0_2': (3, 3, 64, 64), 'W_upconv2_1': (2, 2, 512, 256), 'b__conv2_2': (256,),
                      'b__conv2_1': (256,), 'gamma_conv3_2': (512,), 'W_conv1_1': (3, 3, 64, 128), 'W_conv1_2': (3, 3, 128, 128),
                      'b__conv3_2': (512,), 'gamma_conv4_2': (1024,), 'b__conv3_1': (512,), 'b__conv0_1': (64,),
                      'W__conv1_2': (3, 3, 128, 128), 'W__conv1_1': (3, 3, 256, 128), 'b__conv0_2': (64,)}

    # mean_dict_shape = {'mean_3': (512,), 'var_3': (512,), 'mean_1': (128,), 'var_1': (128,), 'mean_4': (1024,), 'var_4': (1024,),
    #                    'mean_0': (64,), 'var_2': (256,), 'mean_2': (256,), 'var_0': (64,)}
    f = open('./mean_var.pickle','rb')
    mean_dict = pickle.load(f)
    f.close()

    pred = model(data, var_dict_shape, mean_dict, keep_prob, spatial_keep_prob)

    return pred

