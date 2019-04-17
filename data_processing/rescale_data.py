from data_processing.dataset_utils import RescaleData, DataInfo


import numpy as np
import cv2

import argparse
import os

parser = argparse.ArgumentParser(description='process labels')
parser.add_argument('--data_dir',
                    required=True,
                    type=str,
                    help='path to the data')
args = parser.parse_args()


def main():
    data_dir = os.listdir(args.data_dir)
    info = DataInfo()
    for data_id in data_dir:
        parent_dir = os.path.join(args.data_dir, data_id)
        print(parent_dir)
        os.mkdir(os.path.join(parent_dir, 'zoomed_height_map_color'))
        os.mkdir(os.path.join(parent_dir, 'zoomed_height_map_depth'))
        os.mkdir(os.path.join(parent_dir, 'zoomed_label'))
        background_depth = cv2.imread(os.path.join(parent_dir, 'background_depth.png'), cv2.IMREAD_ANYDEPTH)
        with open(os.path.join(parent_dir, 'file_name.txt'), 'r') as f:
            with open(os.path.join(parent_dir, 'zoomed_file_name.txt'), 'w') as f_zoom:
                file_names = f.readlines()
                for file_name in file_names:
                    file_name = file_name[:-2] if file_name[-2:] == '\r\n' else file_name[:-1]
                    print(file_name)
                    color = cv2.imread(os.path.join(parent_dir, 'color', file_name + '.png'))
                    depth = cv2.imread(os.path.join(parent_dir, 'depth', file_name + '.png'), cv2.IMREAD_ANYDEPTH)
                    try:
                        grasp_labels = np.loadtxt(os.path.join(parent_dir, 'label', file_name + '.good.txt'))
                        label_flag = 'good'
                    except IOError:
                        grasp_labels = np.loadtxt(os.path.join(parent_dir, 'label', file_name + '.bad.txt'))
                        label_flag = 'bad'
                    rd = RescaleData(color, depth, background_depth, grasp_labels, info)
                    for i in range(7):
                        factor = np.random.uniform(0.20/info.camera_height, 1.0)
                        crop_color, crop_depth, label, label_points, object_points = rd.get_zoomed_data(factor)
                        cv2.imwrite(os.path.join(parent_dir, 'zoomed_height_map_color',
                                                 file_name + '.{:0.7f}.png'.format(factor)), crop_color)
                        cv2.imwrite(os.path.join(parent_dir, 'zoomed_height_map_depth',
                                                 file_name + '.{:0.7f}.png'.format(factor)), crop_depth)
                        cv2.imwrite(os.path.join(parent_dir, 'zoomed_label', file_name + '.{:0.7f}.png'.format(factor)),
                                    label)
                        np.savetxt(os.path.join(parent_dir, 'zoomed_label',
                                                file_name + '.{:0.7f}.'.format(factor) + label_flag + '.txt'),
                                   label_points)
                        np.savetxt(os.path.join(parent_dir, 'zoomed_label',
                                                file_name + '.{:0.7f}.'.format(factor) + 'object_points.txt'),
                                   object_points)
                        f_zoom.write(file_name + '.{:0.7f}\n'.format(factor))


if __name__ == '__main__':
    main()


