import tensorflow as tf
import random
import math

import argparse
import os

from data_processing.dataset_utils import *

parser = argparse.ArgumentParser(description='create tensorflow record')
parser.add_argument('--data_path',
                    required=True,
                    type=str,
                    help='Path to data set.')
parser.add_argument('--output',
                    default='data/tfrecords',
                    type=str,
                    help='Path to the output files.')
args = parser.parse_args()

folders = ['color', 'encoded_depth', 'label_map', 'camera_height']


def dict_to_tf_example(file_name):
    """
    Create tfrecord example.
    :param file_name: File path corresponding to the data.
    :return: example: Example of tfrecord.
    """
    with open(os.path.join(args.data_path, folders[0], file_name+'.png'), 'rb') as fid:
        encoded_color = fid.read()
    with open(os.path.join(args.data_path, folders[1], file_name + '.png'), 'rb') as fid:
        encoded_depth = fid.read()
    with open(os.path.join(args.data_path, folders[2], file_name + '.png'), 'rb') as fid:
        encoded_label_map = fid.read()
    # camera_height = float(np.loadtxt(os.path.join(args.data_path, folders[3], file_name + '.txt')))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/color': bytes_feature(encoded_color),
        'image/format': bytes_feature(b'png'),
        'image/encoded_depth': bytes_feature(encoded_depth),
        'image/label': bytes_feature(encoded_label_map),
        # 'image/camera_height': float_feature(camera_height),
    }))
    return example


def main():
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        os.makedirs(os.path.join(args.output, 'pos'))
        os.makedirs(os.path.join(args.output, 'neg'))
    curr_pos_records = len(os.listdir(os.path.join(args.output, 'pos')))
    curr_neg_records = len(os.listdir(os.path.join(args.output, 'neg')))
    curr_records_list = [curr_pos_records, curr_neg_records]
    sets = ['pos', 'neg']
    num_samples_per_record = 10000

    for curr_records, set in zip(curr_records_list, sets):
        with open(os.path.join(args.data_path, set+'.txt'), 'r') as f:
            file_list = f.readlines()
        random.shuffle(file_list)
        num_samples = len(file_list)
        num_records = math.ceil(num_samples / num_samples_per_record)
        for i, record_idx in enumerate(range(curr_pos_records, curr_pos_records + num_records)):
            writer = tf.python_io.TFRecordWriter(os.path.join(args.output,
                                                              set,
                                                              'train_{}_{:04d}.tfrecord'.format(set, record_idx)))
            j = i * num_samples_per_record
            while j < (i + 1) * num_samples_per_record and j < num_samples:
                tf_example = dict_to_tf_example(file_list[j][:-1])
                writer.write(tf_example.SerializeToString())
                j += 1
            writer.close()


if __name__ == '__main__':
    main()

