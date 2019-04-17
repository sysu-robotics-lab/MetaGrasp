from hparams import create_metagrasp_hparams
from network_utils import *
from models import metagrasp
from losses import *

import tensorflow as tf
import tensorflow.contrib.slim as slim

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='network training')
parser.add_argument('--master',
                    default='',
                    type=str,
                    help='BNS name of the TensorFlow master to use')
parser.add_argument('--task_id',
                    default=0,
                    type=int,
                    help='The Task ID. This value is used when training with multiple workers to identify each worker.')
parser.add_argument('--train_log_dir',
                    default='models/metagrasp',
                    type=str,
                    help='Directory where to write event models.')
parser.add_argument('--dataset_dir',
                    default='data/tfrecords',
                    type=str,
                    help='The directory where the datasets can be found.')
parser.add_argument('--save_summaries_steps',
                    default=120,
                    type=int,
                    help='The frequency with which summaries are saved, in seconds.')
parser.add_argument('--save_interval_secs',
                    default=600,
                    type=int,
                    help='The frequency with which the model is saved, in seconds.')
parser.add_argument('--print_loss_steps',
                    default=100,
                    type=int,
                    help='The frequency with which the losses are printed, in steps.')
parser.add_argument('--num_readers',
                    default=2,
                    type=int,
                    help='The number of parallel readers that read data from the dataset.')
parser.add_argument('--num_steps',
                    default=200000,
                    type=int,
                    help='The max number of gradient steps to take during training.')
parser.add_argument('--num_preprocessing_threads',
                    default=4,
                    type=int,
                    help='The number of threads used to create the batches.')
parser.add_argument('--from_metagrasp_checkpoint',
                    default=False,
                    type=bool,
                    help='load checkpoint from metagrasp checkpoint or classification checkpoint.')
parser.add_argument('--checkpoint_dir',
                    default='',
                    type=str,
                    help='The directory where the checkpoint can be found')
args = parser.parse_args()


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    h = create_metagrasp_hparams()
    for path in [args.train_log_dir]:
        if not tf.gfile.Exists(path):
            tf.gfile.MakeDirs(path)
    hparams_filename = os.path.join(args.train_log_dir, 'hparams.json')
    with tf.gfile.FastGFile(hparams_filename, 'w') as f:
        f.write(h.to_json())
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(args.task_id)):
            global_step = tf.train.get_or_create_global_step()
            colors_p, labels_p = get_color_dataset(args.dataset_dir + '/pos',
                                                   args.num_readers,
                                                   args.num_preprocessing_threads,
                                                   h.image_size,
                                                   h.label_size,
                                                   int(h.batch_size/2))
            colors_n, labels_n = get_color_dataset(args.dataset_dir + '/neg',
                                                   args.num_readers,
                                                   args.num_preprocessing_threads,
                                                   h.image_size,
                                                   h.label_size,
                                                   int(h.batch_size/2))
            colors = tf.concat([colors_p, colors_n], axis=0)
            labels = tf.concat([labels_p, labels_n], axis=0)
            net, end_points = metagrasp(colors,
                                        num_classes=3,
                                        num_channels=1000,
                                        is_training=True,
                                        global_pool=False,
                                        output_stride=16,
                                        spatial_squeeze=False,
                                        scope=h.scope)
            loss = create_loss_with_label_mask(net, labels, h.lamb)
            learning_rate = h.learning_rate
            if h.lr_decay_step:
                learning_rate = tf.train.exponential_decay(h.learning_rate,
                                                           tf.train.get_or_create_global_step(),
                                                           decay_steps=h.lr_decay_step,
                                                           decay_rate=h.lr_decay_rate,
                                                           staircase=True)
            tf.summary.scalar('Learning_rate', learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = slim.learning.create_train_op(loss, optimizer)
            add_summary(colors, labels, end_points, loss, h)
            summary_op = tf.summary.merge_all()
            if not args.from_metagrasp_checkpoint:
                variable_map = restore_from_classification_checkpoint(
                    scope=h.scope,
                    model_name=h.model_name,
                    checkpoint_exclude_scopes=['prediction'])
                init_saver = tf.train.Saver(variable_map)

                def initializer_fn(sess):
                    init_saver.restore(sess, os.path.join(args.checkpoint_dir, h.model_name+'.ckpt'))
                    tf.logging.info('Successfully load pretrained checkpoint.')

                init_fn = initializer_fn

            else:
                variable_map = restore_map()
                init_saver = tf.train.Saver(variable_map)

                def initializer_fn(sess):
                    init_saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
                    tf.logging.info('Successfully load pretrained checkpoint.')

                init_fn = initializer_fn

            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False)
            session_config.gpu_options.allow_growth = True
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=args.save_interval_secs,
                                   max_to_keep=100)

            slim.learning.train(train_op,
                                logdir=args.train_log_dir,
                                master=args.master,
                                global_step=global_step,
                                session_config=session_config,
                                init_fn=init_fn,
                                summary_op=summary_op,
                                number_of_steps=args.num_steps,
                                startup_delay_steps=15,
                                save_summaries_secs=args.save_summaries_steps,
                                saver=saver)


if __name__ == '__main__':
    main()
