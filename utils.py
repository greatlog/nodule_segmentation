from skimage.measure import label,regionprops
import os
import tensorflow as tf

def find_bounding_box(input):
    bbox=[]
    for i in range(input.shape[0]):
        labeled_reg = label(input[i,:,:],8)
        property = regionprops(labeled_reg)
        bbox_ = []
        for i in range(len(property)):
            cx,cy = property[i].centroid
            b = property[i].bbox
            d = max(abs(b[0]-b[2]),abs(b[1]-b[3]))
            if d>5:
                bbox_.append([cx,cy,d])
        bbox.append(bbox_)
    return bbox

def generate_list(flags):
    list1_ = os.listdir(os.path.join(flags.train_dir, 'image_jpg'))
    list2_ = os.listdir(os.path.join(flags.train_dir, 'mask_jpg'))
    list1 = [os.path.join(flags.train_dir, 'image_jpg', _) for _ in list1_ if _.split('.')[-1] == 'jpg']
    list2 = [os.path.join(flags.train_dir, 'mask_jpg', _) for _ in list2_ if _.split('.')[-1] == 'jpg']
    list1.sort()
    list2.sort()

    length = len(list1)

    tr_list1 = tf.convert_to_tensor(list1[0:int(0.8 * length)],tf.string)
    tr_list2 = tf.convert_to_tensor(list2[0:int(0.8 * length)],tf.string)
    tr_list = [tr_list1,tr_list2]

    ts_list1 = tf.convert_to_tensor(list1[int(0.8 * length):length - 1],tf.string)
    ts_list2 = tf.convert_to_tensor(list2[int(0.8 * length):length - 1],tf.string)
    ts_list = [ts_list1,ts_list2]

    if_list1 = tf.convert_to_tensor(list1,tf.string)
    if_list2 = tf.convert_to_tensor(list2,tf.string)
    if_list = [if_list1,if_list2]

    return tr_list,ts_list,if_list,length

