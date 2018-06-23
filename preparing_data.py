import os
from pydicom import dicomio
from PIL import Image
import tensorflow as tf
from preprocess import *

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


root_dir1 = '/data/20180507/noduledata/DOI'
root_dir2 = '/data/20180507/noduledata/MASK1'

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        list1 = os.listdir(root_dir1)
        for dir1 in list1:
            second_dir = os.path.join(root_dir1, dir1)
            list2 = os.listdir(second_dir)
            list2.sort()
            for dir2 in list2:
                third_dir = os.path.join(second_dir, dir2)
                list3 = os.listdir(third_dir)
                list3.sort()
                for dir3 in list3:
                    fourth_dir = os.path.join(third_dir, dir3)
                    file = os.listdir(fourth_dir)
                    file.sort()
                    if len(file) > 10:
                        print("processing image", dir1)
                        count = 0
                        imgs, mask_2, spc = mask_extractor(fourth_dir)
                        list4 = os.listdir(fourth_dir)
                        original_image_list = [tmp for tmp in list4 if tmp.split('.')[-1] == 'dcm']
                        original_image_list.sort()
                        sliceim, _ = mask_processing(imgs, mask_2, spc)
                        mask_path = os.path.join(root_dir2, dir1, dir2, dir3)
                        mask_file = os.listdir(mask_path)
                        mask_file.sort()
                        for f in mask_file:
                            name = f.split('_')
                            if len(name) > 1:
                                original_name = name[-1]
                                ds = dicomio.read_file(os.path.join(fourth_dir,original_name))
                                z = ds.InstanceNumber
                                slice_id = z-1
                                #
                                label_path = os.path.join(mask_path, f)
                                print("processing label", f)
                                label, _, _ = load_itk(label_path)
                                #label = np.resize(label, [300, 300])
                                label = np.squeeze(label, 0)
                                label_img = Image.fromarray(label)
                                label_img.save("/data/model_weights/luozx/image_jpg/{}_{}.jpg".format(dir1, f))
                                label = label.tostring()

                                img = imgs[slice_id].astype(np.float32)
                                img_img = Image.fromarray(img)
                                img_img.save("/data/model_weights/luozx/mask_jpg/{}_{}.jpg".format(dir1, f))
                                img = img.tostring()

                                feature = {'data': bytes_feature(img),
                                   'label':bytes_feature(label),
                                   'cor_x':bytes_feature(np.array(z).tostring())}

                                example = tf.train.Example(features=tf.train.Features(feature=feature))
                                writer = tf.python_io.TFRecordWriter(
                                    "/data/model_weights/luozx/nodule_tfrecords/{}bactch{}.tfrecords".format(dir1, count))
                                writer.write(example.SerializeToString())
                                writer.close()
                                count = count + 1

                                print(count)





