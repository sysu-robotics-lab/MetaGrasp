from data_processing.data_processor import DataProcessor

import numpy as np
import cv2

import argparse
import random
import os

parser = argparse.ArgumentParser(description='process data')
parser.add_argument('--data_dir',
                    required=True,
                    type=str,
                    help='path to the data')
parser.add_argument('--output',
                    required=True,
                    type=str,
                    help='path to save the process data')
args = parser.parse_args()


def main():
    if not os.path.exists(args.output):
        os.mkdir(args.output)
        os.mkdir(os.path.join(args.output, 'color'))
        os.mkdir(os.path.join(args.output, 'depth'))
        os.mkdir(os.path.join(args.output, 'encoded_depth'))
        os.mkdir(os.path.join(args.output, 'label_map'))
        os.mkdir(os.path.join(args.output, 'camera_height'))
    f_p = open(os.path.join(args.output, 'pos.txt'), 'w')
    f_n = open(os.path.join(args.output, 'neg.txt'), 'w')
    dp = DataProcessor()

    # load data
    data_dir = os.listdir(args.data_dir)
    random.shuffle(data_dir)
    for data_id in data_dir:
        parent_dir = os.path.join(args.data_dir, data_id)
        print(parent_dir)
        # b_depth_height_map = cv2.imread(os.path.join(parent_dir, 'crop_background_depth.png'),
        #                                 cv2.IMREAD_ANYDEPTH).astype(np.float32)
        label_files = os.listdir(os.path.join(parent_dir, 'label'))
        with open(os.path.join(parent_dir, 'file_name.txt'), 'r') as f:
            file_names = f.readlines()
        for file_name in file_names:
            file_name = file_name[:-2] if file_name[-2:] == '\r\n' else file_name[:-1]
            print(file_name)
            color = cv2.imread(os.path.join(parent_dir, 'height_map_color', file_name+'.png'))
            depth = cv2.imread(os.path.join(parent_dir, 'height_map_depth', file_name+'.png'),
                               cv2.IMREAD_ANYDEPTH).astype(np.float32)
            # diff_depth = dp.get_diff_depth(depth, b_depth_height_map)
            # pad color and depth images
            pad_size = 44
            pad_color = np.ones((288, 288, 3), dtype=np.uint8) * 7
            pad_color[pad_size:pad_size + 200, pad_size:pad_size + 200, :] = color
            pad_depth = np.ones((288, 288), dtype=np.uint16) * 7
            pad_depth[pad_size:pad_size + 200, pad_size:pad_size + 200] = depth
            background_depth_value = np.argmax(np.bincount(depth.astype(np.int).flatten()))
            camera_height = background_depth_value / 1000.0
            neglect_points = np.loadtxt(os.path.join(parent_dir, 'label', file_name+'.object_points.txt')) + pad_size
            if file_name+'.good.txt' in label_files:
                good_pixel_labels = np.loadtxt(os.path.join(parent_dir, 'label', file_name+'.good.txt')) + pad_size
                grasp_centers = dp.get_grasp_center(good_pixel_labels)
                angle_indices = dp.get_grasp_angle(good_pixel_labels[0])
                for angle_idx in angle_indices:
                    quantified_angle = 22.5 * angle_idx
                    for i, angle in enumerate(np.arange(quantified_angle-5, quantified_angle+5, 1)):
                        grasp_label = np.zeros((36, 36, 3), dtype=np.uint8)  # bgr for opencv
                        grasp_label[..., 0] = 255
                        rotated_neglect_points = dp.rotate(neglect_points, (144, 144), (angle / 360.0) * np.pi * 2)
                        rotated_neglect_points = np.round(rotated_neglect_points / 8.0).astype(np.int)
                        grasp_label[rotated_neglect_points[:, 0], rotated_neglect_points[:, 1], 0] = 0
                        grasp_label = cv2.medianBlur(grasp_label, 3)
                        rotated_grasp_centers = dp.rotate(grasp_centers, (144, 144), (angle / 360.0) * np.pi * 2)
                        rotated_grasp_centers = np.round(rotated_grasp_centers / 8.0).astype(np.int)
                        grasp_label[rotated_grasp_centers[:, 0], rotated_grasp_centers[:, 1], 1] = 255
                        mtx = cv2.getRotationMatrix2D((144, 144), angle, 1)
                        rotated_color = cv2.warpAffine(pad_color, mtx, (288, 288),
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(7, 7, 7))
                        rotated_depth = cv2.warpAffine(pad_depth, mtx, (288, 288),
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(7, 7, 7))
                        encoded_depth = dp.encode_depth(rotated_depth)
                        cv2.imwrite(os.path.join(args.output, 'label_map',
                                                 data_id + '-' + file_name + '-{}-{:02d}.png'.format(angle_idx, i)),
                                    grasp_label)
                        cv2.imwrite(os.path.join(args.output, 'color',
                                                 data_id + '-' + file_name + '-{}-{:02d}.png'.format(angle_idx, i)),
                                    rotated_color)
                        cv2.imwrite(os.path.join(args.output, 'depth',
                                                 data_id + '-' + file_name + '-{}-{:02d}.png'.format(angle_idx, i)),
                                    rotated_depth)
                        cv2.imwrite(os.path.join(args.output, 'encoded_depth',
                                                 data_id + '-' + file_name + '-{}-{:02d}.png'.format(angle_idx, i)),
                                    encoded_depth)
                        np.savetxt(os.path.join(args.output, 'camera_height',
                                                data_id + '-' + file_name + '-{}-{:02d}.txt'.format(angle_idx, i)),
                                   np.array([camera_height * 1000.0]))
                        f_p.write(data_id + '-' + file_name + '-{}-{:02d}\n'.format(angle_idx, i))
            else:
                bad_pixel_labels = np.loadtxt(os.path.join(parent_dir, 'label', file_name + '.bad.txt')) + pad_size
                grasp_centers = dp.get_grasp_center(bad_pixel_labels)
                angle_indices = dp.get_grasp_angle(bad_pixel_labels[0])
                for angle_idx in angle_indices:
                    quantified_angle = 22.5 * angle_idx
                    for i, angle in enumerate(np.arange(quantified_angle - 5, quantified_angle + 5, 1)):
                        grasp_label = np.zeros((36, 36, 3), dtype=np.uint8)  # bgr for opencv
                        grasp_label[..., 0] = 255
                        rotated_neglect_points = dp.rotate(neglect_points, (144, 144), (angle / 360.0) * np.pi * 2)
                        rotated_neglect_points = np.round(rotated_neglect_points / 8.0).astype(np.int)
                        grasp_label[rotated_neglect_points[:, 0], rotated_neglect_points[:, 1], 0] = 0
                        grasp_label = cv2.medianBlur(grasp_label, 3)
                        rotated_grasp_centers = dp.rotate(grasp_centers, (144, 144), (angle / 360.0) * np.pi * 2)
                        rotated_grasp_centers = np.round(rotated_grasp_centers / 8.0).astype(np.int)
                        grasp_label[rotated_grasp_centers[:, 0], rotated_grasp_centers[:, 1], 2] = 255
                        mtx = cv2.getRotationMatrix2D((144, 144), angle, 1)
                        rotated_color = cv2.warpAffine(pad_color, mtx, (288, 288),
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(7, 7, 7))
                        rotated_depth = cv2.warpAffine(pad_depth, mtx, (288, 288),
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(7, 7, 7))
                        encoded_depth = dp.encode_depth(rotated_depth)
                        cv2.imwrite(os.path.join(args.output, 'label_map',
                                                 data_id + '-' + file_name + '-{}-{:02d}.png'.format(angle_idx, i)),
                                    grasp_label)
                        cv2.imwrite(os.path.join(args.output, 'color',
                                                 data_id + '-' + file_name + '-{}-{:02d}.png'.format(angle_idx, i)),
                                    rotated_color)
                        cv2.imwrite(os.path.join(args.output, 'depth',
                                                 data_id + '-' + file_name + '-{}-{:02d}.png'.format(angle_idx, i)),
                                    rotated_depth)
                        cv2.imwrite(os.path.join(args.output, 'encoded_depth',
                                                 data_id + '-' + file_name + '-{}-{:02d}.png'.format(angle_idx, i)),
                                    encoded_depth)
                        np.savetxt(os.path.join(args.output, 'camera_height',
                                                data_id + '-' + file_name + '-{}-{:02d}.txt'.format(angle_idx, i)),
                                   np.array([camera_height * 1000.0]))
                        f_n.write(data_id + '-' + file_name + '-{}-{:02d}\n'.format(angle_idx, i))
    f_p.close()
    f_n.close()


if __name__ == '__main__':
    main()

