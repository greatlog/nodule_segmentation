from unet2d import build_UNet2D
import tensorflow as tf
import collections


def nodule_seg(data, label, weight, flags):
    net = collections.namedtuple('Network',
                                 'loss,mask,train_op,learning_rate,global_step,accuracy_front,accuracy_back,label,unweighted_loss')

    with tf.variable_scope('forward',reuse=tf.AUTO_REUSE):
        unet = build_UNet2D(data, flags.epsilon, flags.b_momentum, flags.num_class)

    with tf.variable_scope('build_loss',reuse=tf.AUTO_REUSE):
        convert_label = tf.one_hot(label, depth=flags.num_class, axis=-1)
        flat_logits = tf.reshape(unet,[-1,2])
        flat_labels = tf.reshape(convert_label,[-1,2])
        flat_weights = tf.reshape(weight,[-1,])
        if flags.mode == 'train':
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                               labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, flat_weights)

            loss = tf.reduce_mean(weighted_loss)
            unweighted_loss = tf.reduce_mean(loss_map)
        else:
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                               labels=flat_labels)
            unweighted_loss = tf.reduce_mean(loss_map)

    with tf.variable_scope('compute_output',reuse=tf.AUTO_REUSE):
        prob = tf.arg_max(unet, -1)
        mask = tf.cast(prob, tf.uint8)

    with tf.variable_scope('accuracy',reuse=tf.AUTO_REUSE):
        mask_front = tf.cast(tf.equal(mask, 1), tf.float32)
        mask_back = tf.cast(tf.equal(mask, 0), tf.float32)

        label_front = tf.cast(tf.equal(label, 255), tf.float32)
        label_back = tf.cast(tf.equal(label, 0), tf.float32)

        pp = tf.equal(tf.add(mask_front, label_front), 2)
        bp = tf.equal(tf.add(mask_back, label_back), 2)

        accuracy_front = tf.reduce_sum(tf.cast(pp, tf.float32)) / tf.reduce_sum(label_front)
        accuracy_back = tf.reduce_sum(tf.cast(bp, tf.float32)) / tf.reduce_sum(label_back)

    with tf.variable_scope('learning_rate_and_global_step',reuse=tf.AUTO_REUSE):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_steps,
                                                   flags.decay_rate, staircase=flags.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta)
        train_op = tf.group(optimizer.minimize(loss), incr_global_step)

    return net(loss=loss, train_op=train_op,
               accuracy_front=accuracy_front,
               accuracy_back=accuracy_back,
               learning_rate=learning_rate,
               global_step=global_step,
               unweighted_loss=unweighted_loss,
               label=label,
               mask=mask)
