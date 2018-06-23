import tensorflow as tf
from tensorflow.contrib.layers import repeat,conv2d,max_pool2d
from keras.layers import UpSampling2D

def down_block(input,epsilon,b_momentum,scope,k_size,output):
    with tf.variable_scope(scope):
        conv = repeat(input, 2, conv2d, output, k_size, scope='conv')
        pool = max_pool2d(conv, 2,scope='pool')
        bn = tf.layers.batch_normalization(pool, epsilon=epsilon, axis=-1, momentum=b_momentum,
                                            beta_initializer=tf.zeros_initializer(),
                                            gamma_initializer=tf.constant_initializer(0.2), name='bn')
        return conv, bn

def up_block(input,epsilon,b_momentum,scope,k_size,output,pre_conv):
    with tf.variable_scope(scope):
        up = UpSampling2D()(input)
        conv = repeat(up,2,conv2d,output,k_size)
        bn = tf.layers.batch_normalization(conv, epsilon=epsilon, axis=-1, momentum=b_momentum,
                                           beta_initializer=tf.zeros_initializer(),
                                           gamma_initializer=tf.constant_initializer(0.2), name='bn')
        merge = tf.concat([pre_conv,bn],axis=-1)
        conv = conv2d(merge,output,k_size)
        return conv



def build_UNet2D(input,epsilon,b_momentum,num_class,k_size=3,output = 32):
    conv1,bn1 = down_block(input, epsilon, b_momentum, scope='down1', k_size=3, output=output)
    conv2,bn2 = down_block(bn1, epsilon, b_momentum, scope='down2', k_size=3, output=output * 2)
    conv3,bn3 = down_block(bn2, epsilon, b_momentum, scope='down3', k_size=3, output=output * 2)
    conv4,bn4 = down_block(bn3,epsilon, b_momentum, scope='down4', k_size=3, output=output*4)

    bottom = conv2d(bn4,output*8,k_size,scope='bottom')

    up1 = up_block(bottom,epsilon,b_momentum,'up1',k_size=k_size,output=output*8,pre_conv=conv4)
    up2 = up_block(up1,epsilon,b_momentum,'up2',k_size=k_size,output=output*8,pre_conv=conv3)
    up3 = up_block(up2, epsilon, b_momentum, 'up3', k_size=k_size, output=output * 4, pre_conv=conv2)
    up4 = up_block(up3, epsilon, b_momentum, 'up4', k_size=k_size, output=output * 2, pre_conv=conv1)

    out = conv2d(up4,num_class,k_size,activation_fn=None)

    return out



