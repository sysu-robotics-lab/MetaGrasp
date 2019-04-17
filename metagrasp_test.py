from models import *

import tensorflow as tf
import numpy as np
import cv2

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='network evaluating')
parser.add_argument('--color_dir',
                    default='data/test/000024.png',
                    type=str,
                    help='The directory where the color image can be found.')
parser.add_argument('--output_dir',
                    default='data/output',
                    type=str,
                    help='The directory where the color image can be found.')
parser.add_argument('--checkpoint_dir',
                    default='models/metagrasp',
                    type=str,
                    help='The directory where the checkpoint can be found')
args = parser.parse_args()


def main():
    colors_n = cv2.resize(cv2.imread(args.color_dir), (200, 200))[..., ::-1]
    pad_size = 44
    pad_colors = np.zeros((288, 288, 3), dtype=np.uint8)
    pad_colors[pad_size:pad_size + 200, pad_size:pad_size + 200, :] = colors_n

    colors_p = tf.placeholder(dtype=tf.float32, shape=[1, 288, 288, 3])
    colors = colors_p * tf.random_normal(colors_p.get_shape(), mean=1, stddev=0.01)
    colors = colors / tf.constant([255.0])
    colors = (colors - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])

    net, end_points = metagrasp(colors,
                                num_classes=3,
                                num_channels=1000,
                                is_training=False,
                                global_pool=False,
                                output_stride=16,
                                spatial_squeeze=False,
                                scope='metagrasp')
    probability_map = tf.exp(net) / tf.reduce_sum(tf.exp(net), axis=3, keepdims=True)
    probability_map = tf.image.resize_bilinear(probability_map, [288, 288])
    saver = tf.train.Saver()
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
    print('Successfully loading model: {}.'.format(tf.train.latest_checkpoint(args.checkpoint_dir)))
    sess.run(tf.local_variables_initializer())

    outputs_a = []
    outputs_a_plus_c = []
    for i in range(16):
        mtx = cv2.getRotationMatrix2D((144, 144), 22.5 * i, 1)
        rotated_colors = cv2.warpAffine(pad_colors, mtx, (288, 288))

        output = sess.run(probability_map,
                          feed_dict={colors_p: np.expand_dims(rotated_colors, 0).astype(np.float32)})
        outputs_a.append(output[..., 1])  # extract green channel
        outputs_a_plus_c.append((rotated_colors*0.7 + np.squeeze(output)*255.0*0.3).astype(np.uint8))

        cv2.imwrite(os.path.join(args.output_dir, 'rotated_colors_{}.png'.format(i)),
                    rotated_colors[..., ::-1])
        cv2.imwrite(os.path.join(args.output_dir, 'output_{}.png'.format(i)),
                    (np.squeeze(output)*255.0).astype(np.uint8)[..., ::-1])
    outputs = np.concatenate(outputs_a, axis=0)
    threshold = np.max(outputs) - 0.001
    for idx, h, w in zip(*np.where(outputs >= threshold)):
        cv2.circle(outputs_a_plus_c[idx], (w, h), 1, color=(0, 255, 0), thickness=5)
    vis_map = np.concatenate(
        tuple([np.concatenate(tuple(outputs_a_plus_c[4 * i:4 * (i + 1)]), axis=1) for i in range(4)]),
        axis=0)
    cv2.imshow('visualization_map', vis_map[..., ::-1])
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(args.output_dir, 'visualization_map.png'), vis_map[..., ::-1])

    sess.close()


if __name__ == '__main__':
    main()

