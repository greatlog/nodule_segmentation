from model_2 import nodule_seg
#from data_loader import *
from dataset import *
from utils import *
import time
import pandas as pd

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('train_dir', None, 'the directory of training data')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('test_dir', None, 'The directory of the test data')

Flags.DEFINE_integer('label_size', 324, 'The size of label')
Flags.DEFINE_integer('size', 512, 'The size of imgae')
Flags.DEFINE_integer('epochs', 50, 'the number of epochs')

Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
Flags.DEFINE_float('epsilon', None, 'the value of epsilon in batchnormalization')
Flags.DEFINE_float('b_momentum', None, 'the value of momentum in batchnormalization')
Flags.DEFINE_integer('num_class', 2, 'the number of class in the CT to predict')

Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_boolean('pre_trained_model', False,
                     'If set True, the weight will be loaded but the global_step will still '
                     'be 0. If set False, you are going to continue the training. That is, '
                     'the global_step will be initiallized from the checkpoint, too')
Flags.DEFINE_string('pretrained_unet', '/data/model_weights/luozx/unet_model_50000/', 'pretrained_unet')
Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')

Flags.DEFINE_float('learning_rate', 0.00001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_steps', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('momentum', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')

flags = Flags.FLAGS

# Check the output_dir is given
if flags.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(flags.output_dir):
    os.mkdir(flags.output_dir)

# Check the summary directory to save the event
if not os.path.exists(flags.summary_dir):
    os.mkdir(flags.summary_dir)

tr_list,ts_list,if_list = generate_list(flags)

tr_iterator = data_loader(tr_list[0],tr_list[1],flags)
ts_iterator = data_loader(ts_list[0],ts_list[1],flags,1)
if_iterator = data_loader(if_list[0].if_list[1],flags,1)

handle = tf.placeholder(dtype=tf.string,shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, [tf.float32,tf.float32], [flags.data_shape,flags.label_shape])
batch_names, batch_images, batch_labels, batch_weights, batch_ratios, batch_scales = iterator.get_next()

net = nodule_seg(batch_images, batch_labels, batch_weights, flags)

if flags.mode == 'train':

    tf.summary.scalar('loss', net.loss)
    tf.summary.scalar('unweighted_loss', net.unweighted_loss)
    tf.summary.scalar('acc_f', net.accuracy_front)
    tf.summary.scalar('acc_b', net.accuracy_back)
    tf.summary.scalar('learning_rate', net.learning_rate)

    saver = tf.train.Saver(max_to_keep=10)

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='forward')

    weight_initiallizer = tf.train.Saver(var_list2)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Use superviser to coordinate all queue and summary writer
    sv = tf.train.Supervisor(logdir=flags.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:

        if (flags.checkpoint is not None) and (flags.pre_trained_model is False):
            print('Loading model from the checkpoint...')
            checkpoint = tf.train.latest_checkpoint(flags.checkpoint)
            saver.restore(sess, checkpoint)

        elif (flags.checkpoint is not None) and (flags.pre_trained_model is True):
            print('Loading weights from the pre-trained model')
            weight_initiallizer.restore(sess, flags.checkpoint)

        # elif(flags.checkpoint is None):
        #     sess.run(initop)
        train_handle = sess.run(tr_iterator.string_handle())

        print('Optimization starts!!!')
        start = time.time()

        fd_results = open(os.path.join(flags.output_dir, './validation.txt'), 'a+')
        step = 0
        while True:
            try:
                fetches = {
                    'train': net.train_op,
                    'global_step': sv.global_step,
                }

                if ((step + 1) % flags.display_freq) == 0:
                    fetches['loss'] = net.loss
                    fetches['unweighted_loss'] = net.unweighted_loss
                    # fetches['mask'] = net.mask
                    # fetches['label'] = net.label
                    # fetches['ratio'] = data.ratio
                    # fetches['scaler'] = data.scaler
                    fetches['acc_f'] = net.accuracy_front
                    fetches['acc_b'] = net.accuracy_back
                    fetches['global_step'] = net.global_step
                    fetches['learning_rate'] = net.learning_rate

                if ((step + 1) % flags.summary_freq) == 0:
                    fetches['summary'] = sv.summary_op

                results = sess.run(fetches,feed_dict = {handle:train_handle})

                if ((step + 1) % flags.summary_freq) == 0:
                    print('Recording summary!!')
                    sv.summary_writer.add_summary(results['summary'], results['global_step'])

                if ((step + 1) % flags.display_freq) == 0:
                    curr_time = time.time()
                    speed = (curr_time - start) / (step)
                    print('seep:%.4fs per batch' % speed)
                    print('step:%d,weighted_loss:%.4f,unweighted_loss:%.4f,acc_f:%.4f,acc_b:%.4f' % (
                    step, results['loss'], results['unweighted_loss'], results['acc_f'], results['acc_b']))
                    # print(results['ratio'])
                    # print(results['scaler'])
                    # print(results['mask'])
                    # print(results['label'])

                if ((step + 1) % flags.save_freq) == 0:
                    print('Save the checkpoint')
                    saver.save(sess, os.path.join(flags.output_dir, 'model'), global_step=step+1)

                if ((step +1) % flags.val_freq) == 0:
                    print("validation...")

                    val_handle = sess.run(ts_iterator.string_handle())

                    total_loss = 0.0
                    total_acc_f = 0.0
                    total_acc_b = 0.0
                    count = 0.0
                    while True:
                        try:
                            loss, acc_f, acc_b = sess.run([net.loss, net.accuracy_front, net.accuracy_back],
                                                          feed_dict={handle: val_handle})
                            total_loss = total_loss + loss
                            total_acc_f = total_acc_f + acc_f
                            total_acc_b = total_acc_b + acc_b
                            count = count + 1

                        except tf.errors.OutOfRangeError as e:
                            average_acc_f = total_acc_f / count
                            average_acc_b = total_acc_b / count
                            average_loss = total_loss / count

                            print("average_loss: %.4f\n" % average_loss)

                            fd_results.write("epochs: d\n" % step)
                            fd_results.write("average_acc_f: %.4f\n" % average_acc_f)
                            fd_results.write("average_acc_f: %.4f\n" % average_acc_b)
                            fd_results.write("average_loss: %.4f\n" % average_loss)
                            break

                    print('validation Done!!')

                step = step + 1
            except tf.errors.OutOfRangeError as e:
                break


        print('Optimization done!!!!!!!!!!!!')

elif flags.mode == 'test':

    if flags.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='forward')
    weight_initiallizer = tf.train.Saver(var_list)
    print('Evaluation starts!!')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=flags.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        ts_handle = sess.run(ts_iterator.string_handle())
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, flags.checkpoint)

        results_name = os.path.join(flags.output_dir, './accuracy_eval.txt')
        if os.path.exists(results_name):
            os.remove(results_name)

        fd_results = open(os.path.join(flags.output_dir, './accuracy_eval.txt'), 'a+')
        print('Evaluation starts!!')

        total_loss = 0.0
        total_acc_f = 0.0
        total_acc_b = 0.0
        count = 0.0
        while True:
            try:
                loss, acc_f, acc_b = sess.run([net.loss, net.accuracy_front, net.accuracy_back],
                                              feed_dict={handle:ts_handle})
                total_loss = total_loss + loss
                total_acc_f = total_acc_f + acc_f
                total_acc_b = total_acc_b + acc_b
                fd_results.write('loss:%.4f,acc_f%.4f,acc_b%.4f\n' % (loss, acc_f, acc_b))
                print('loss:%.4f,acc_f%.4f,acc_b%.4f\n' % (loss, acc_f, acc_b))
                count = count + 1
            except tf.errors.OutOfRangeError as e:
                average_acc_f = total_acc_f / count
                average_acc_b = total_acc_b / count
                average_loss = total_loss / count

                print("average_acc_f: %.4f\n" % average_acc_f)
                print("average_acc_b: %.4f\n" % average_acc_b)
                print("average_loss: %.4f\n" % average_loss)

                fd_results.write("average_acc_f: %.4f\n" % average_acc_f)
                fd_results.write("average_acc_b: %.4f\n" % average_acc_b)
                fd_results.write("average_loss: %.4f\n" % average_loss)
                break

        print('Evaluation Done!!')

elif flags.mode == 'inference':

    if flags.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='forward')
    weight_initiallizer = tf.train.Saver(var_list)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=flags.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        if_handle = sess.run(if_iterator)
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, flags.checkpoint)

        results_name = os.path.join(flags.output_dir, './accuracy_eval.txt')
        if os.path.exists(results_name):
            os.remove(results_name)

        fd_results = open(os.path.join(flags.output_dir, './accuracy_eval.txt'), 'a+')
        print('Evaluation starts!!')

        fetches = {
            "img_list": batch_names,
            "mask": net.mask,
            "label": batch_labels,
            "loss": net.loss,
            "acc_f": net.accuracy_front,
            "acc_b": net.accuracy_back,
        }

        total_loss = 0.0
        total_acc_f = 0.0
        total_acc_b = 0.0
        count = 0.0
        pre_csv_file = os.path.join(flags.output_dir, 'pre_result.csv')
        if os.path.exists(pre_csv_file):
            os.remove(pre_csv_file)
        true_csv_file = os.path.join(flags.output_dir, 'true_result.csv')
        if os.path.exists(true_csv_file):
            os.remove(true_csv_file)
        df1 = pd.DataFrame(columns=('image_path', 'cor_x', 'cor_y', 'diameter'))
        df2 = pd.DataFrame(columns=('image_path', 'cor_x', 'cor_y', 'diameter'))
        count1 = 0
        count2 = 0
        while True:
            try:
                results = sess.run(fetches,feed_dict={handle:if_handle})
                total_loss = total_loss + results["loss"]
                total_acc_b = total_acc_b + results["acc_f"]
                total_acc_f = total_acc_f + results["acc_b"]
                fd_results.write(
                    'loss:%.4f,acc_f%.4f,acc_b%.4f\n' % (results["loss"], results["acc_f"], results["acc_b"]))
                print('loss:%.4f,acc_f%.4f,acc_b%.4f\n' % (results["loss"], results["acc_f"], results["acc_b"]))
                count = count + 1

                pre_bbox_ = find_bounding_box(results["mask"])
                true_bbox_ = find_bounding_box(results["label"])

                for j in range(flags.batch_size):
                    for i in range(len(pre_bbox_[j])):
                        df1.loc[count1] = [results["img_list"][j], pre_bbox_[j][i][0], pre_bbox_[j][i][1],
                                           pre_bbox_[j][i][2]]
                        count1 = count1 + 1

                    for i in range(len(true_bbox_[j])):
                        df2.loc[count] = [results["img_list"][j], true_bbox_[j][i][0], true_bbox_[j][i][1],
                                          true_bbox_[j][i][2]]
                        count2 = count2 + 1

            except tf.errors.OutOfRangeError as e:
                average_acc_f = total_acc_f / count
                average_acc_b = total_acc_b / count
                average_loss = total_loss / count

                df1.to_csv(pre_csv_file, index=False, sep=',')
                df2.to_csv(true_csv_file, index=False, sep=',')

                fd_results.write("average_acc_b: %.4f\n" % average_acc_b)
                fd_results.write("average_acc_f: %.4f\n" % average_acc_f)
                fd_results.write("average_loss: %.4f\n" % average_loss)
                break

        print('Evaluation Done!!')