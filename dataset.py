import tensorflow as tf

def data_loader(data_list,label_list,flags,epochs=None):

    def _compute_weight(data_name,data,label):
        def compute_weight(ratio):
            weight_scaler = (tf.constant(5, dtype='float32') - ratio) / (ratio)
            return weight_scaler

        def non_zero_weight(scaler, mask_front, mask_back):
            return ((scaler * mask_front) + mask_back)

        def zero_weight(mask_back):
            return mask_back

        mask_front = tf.cast(tf.equal(label, 255), tf.float32)
        mask_back = tf.cast(tf.equal(label, 0), tf.float32)
        ratio = tf.cast(tf.reduce_sum(mask_front) / (flags.size * flags.size), dtype='float32')
        is_zero = tf.equal(ratio, 0.0)
        scaler = compute_weight(ratio)

        weight_matrix = tf.cond(is_zero, lambda: zero_weight(mask_back),
                                lambda: non_zero_weight(scaler, mask_front, mask_back))
        return data_name,data,label,weight_matrix, ratio, scaler

    def parse_jpg(data_name,label_name):
        q = tf.read_file([data_name,label_name])
        data = tf.image.decode_jpeg(q[0])
        data = tf.reshape(data,flags.data_shape)
        label = tf.image.decode_jpeg(q[1])
        label = tf.reshape(label,flags.label_shape)
        return data_name,data,label

    dataset = tf.data.Dataset.from_tensor_slices((data_list,label_list))
    dataset = dataset.map(parse_jpg,num_parallel_calls=8)
    dataset = dataset.map(_compute_weight,num_parallel_calls=8)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(flags.batch_size)

    iterator = dataset.make_initializable_iterator()

    return iterator





