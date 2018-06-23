import os
import tensorflow as tf


class data_loader():
    def __init__(self, flags):

        if flags.train_dir == 'None':
            raise ValueError('train data directory is not provided')

        if not os.path.exists(flags.train_dir):
            raise ValueError('train data directory is not found')

        self.batch_size = flags.batch_size
        self.size = flags.size

        self.reader = tf.WholeFileReader()

        list1_ = os.listdir(os.path.join(flags.train_dir, 'image_jpg'))
        list2_ = os.listdir(os.path.join(flags.train_dir, 'mask_jpg'))
        list1 = [os.path.join(flags.train_dir, 'image_jpg', _) for _ in list1_ if _.split('.')[-1] == 'jpg']
        list2 = [os.path.join(flags.train_dir, 'mask_jpg', _) for _ in list2_ if _.split('.')[-1] == 'jpg']
        list1.sort()
        list2.sort()

        length = len(list1)

        if flags.mode == 'train':
            list1_ = list1[0:int(0.8 * length)]
            list2_ = list2[0:int(0.8 * length)]
        elif flags.mode == 'test':
            list1_ = list1[int(0.8 * length):length-1]
            list2_ = list2[int(0.8 * length):length-1]
        else:
            list1_ = list1
            list2_ = list2

        if flags.mode == 'inference':
            self.img_list, self.data, self.label, self.weight, self.ratio = self.batch(list1, list2, flags)
        else:
            self.data, self.label, self.weight, self.ratio, self.scaler = self.batch(list1, list2, flags)

        print('trian data loader is finished,data shape is', self.data.shape, 'label shape is', self.label.shape,
              'weight shape is', self.weight.shape)

    def batch(self, list1, list2, flags):

        filename = tf.train.slice_input_producer([list1, list2], num_epochs=1)

        sample = tf.read_file(filename[0])
        mask = tf.read_file(filename[1])

        sample = tf.image.decode_jpeg(sample)
        mask = tf.image.decode_jpeg(mask)

        data = tf.cast(tf.reshape(sample, [self.size, self.size, 1]), tf.float32)
        label_ = tf.reshape(mask, [self.size, self.size, 1])
        label = tf.squeeze(tf.image.resize_image_with_crop_or_pad(label_, flags.label_size, flags.label_size), -1)

        weight_matrix, ratio, scaler = self._compute_weight(label, flags)
        weight_matrix = tf.expand_dims(weight_matrix, -1)

        if flags.mode == 'inference':
            img_list, img_batch, label_batch, weight_batch, ratio_batch = tf.train.batch(
                [filename[0], data, label, weight_matrix, ratio], batch_size=self.batch_size,
                shapes=[[],
                        [self.size, self.size, 1],
                        [flags.label_size, flags.label_size],
                        [flags.label_size, flags.label_size, 1], []],
                capacity=self.batch_size * 20, num_threads=8,
                enqueue_many=False)
            return img_list, img_batch, label_batch, weight_batch, ratio_batch
        else:

            data_batch, label_batch, weight_batch, ratio_batch, scaler_batch = tf.train.batch(
                [data, label, weight_matrix, ratio, scaler],
                batch_size=self.batch_size,
                shapes=[[self.size, self.size, 1],
                        [flags.label_size, flags.label_size],
                        [flags.label_size, flags.label_size, 1], [], []],
                capacity=self.batch_size * 500, num_threads=8,
                enqueue_many=False)

            print(data_batch.shape, label_batch.shape)

            return data_batch, label_batch, weight_batch, ratio_batch, scaler_batch

    def _compute_weight(self, mask, flags):
        def compute_weight(ratio):
            weight_scaler = (tf.constant(5, dtype='float32') - ratio) / (ratio)
            return weight_scaler

        def non_zero_weight(scaler, mask_front, mask_back):
            return ((scaler * mask_front) + mask_back)

        def zero_weight(mask_back):
            return mask_back

        mask_front = tf.cast(tf.equal(mask, 255), tf.float32)
        mask_back = tf.cast(tf.equal(mask, 0), tf.float32)
        ratio = tf.cast(tf.reduce_sum(mask_front) / (flags.size * flags.size), dtype='float32')
        is_zero = tf.equal(ratio, 0.0)
        scaler = compute_weight(ratio)

        weight_matrix = tf.cond(is_zero, lambda: zero_weight(mask_back),
                                lambda: non_zero_weight(scaler, mask_front, mask_back))
        return weight_matrix, ratio, scaler

    def getlist(self,list1,list2,st,ed):
        return list1[st:ed],list2[st:ed]



