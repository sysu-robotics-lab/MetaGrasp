import tensorflow as tf
import numpy as np
import cv2


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_examples_list(path):
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(',') for line in lines]


class DataInfo(object):
    camera_height = 0.57  # camera height, 0.57m
    original_image_height = 480  # the height of raw image captured from the camera
    original_image_width = 640  # the width of raw image captured from the camera
    top_left_corner_h = 45  # the first height index of the original image from which the crop image is cropped.
    top_left_corner_w = 119  # the first width index of the original image from which the crop image is cropped.
    crop_image_size = 390  # crop image size
    resize_image_size = 200 # resize image size


class RescaleData(object):
    def __init__(self, color, depth, background_depth, grasp_labels, info=DataInfo()):
        self.info = info

        self.color = color
        self.diff_depth = background_depth - depth
        self.grasp_labels = grasp_labels
        self.grasp_center = self.get_grasp_center(self.grasp_labels)
        self.grasp_angle = self.get_grasp_angle(self.grasp_labels[0])
        self.gripper_width = self.get_gripper_width(self.grasp_labels[0])

    def get_zoomed_data(self, factor):
        zoomed_size = int(self.info.crop_image_size * factor)
        keep = zoomed_size / 7
        half = zoomed_size / 2
        top, down, left, right = 1, 0, 1, 0
        keep1 = keep2 = keep
        while top > down:
            top = self.grasp_center[0] - half + keep1 if self.grasp_center[0] > zoomed_size else half
            down = self.grasp_center[0] + half - keep1 if self.info.original_image_height - self.grasp_center[0] > zoomed_size else self.info.original_image_height - half - zoomed_size % 2
            keep1 /= 2
        while left > right:
            left = self.grasp_center[1] - half + keep2 if self.grasp_center[1] > zoomed_size else half
            right = self.grasp_center[1] + half - keep2 if self.info.original_image_width - self.grasp_center[1] > zoomed_size else self.info.original_image_width - half - zoomed_size % 2
            keep2 /= 2
        image_center = np.array([np.random.randint(top, down+1, dtype=int), np.random.randint(left, right+1
                                                                                              , dtype=int)])
        crop_color, crop_depth, object_points = self.zoom(image_center, zoomed_size, factor)
        grasp_point = self.grasp_center - (image_center - half)
        grasp_point = (grasp_point * 200.0 / zoomed_size).astype(np.int)
        label_points = self.get_label(grasp_point, self.grasp_angle, self.gripper_width, factor)
        label = self.draw_label(label_points, 200, 200)
        return crop_color, crop_depth, label, label_points, object_points

    def zoom(self, center, zoomed_size, factor):
        half = zoomed_size / 2
        background_depth = int(self.info.camera_height * 1000 * factor)
        crop_color = self.color[center[0] - half:center[0] + half + zoomed_size % 2, center[1] - half:center[1] + half + zoomed_size % 2, :]
        crop_diff_depth = self.diff_depth[center[0] - half:center[0] + half + zoomed_size % 2, center[1] - half:center[1] + half + zoomed_size % 2]
        crop_depth = background_depth - crop_diff_depth
        crop_color = cv2.resize(crop_color, (200, 200))
        crop_depth = cv2.resize(crop_depth, (200, 200))
        points = np.stack(np.where(cv2.resize(crop_diff_depth, (200, 200)) > 0), axis=0).T

        return crop_color, crop_depth, points

    def get_grasp_center(self, grasp_labels):
        row, _ = grasp_labels.shape
        grasp_center = np.empty((row, 2), dtype=np.float32)
        grasp_center[:, 0] = (grasp_labels[:, 0] + grasp_labels[:, 2]) / 2.0
        grasp_center[:, 1] = (grasp_labels[:, 1] + grasp_labels[:, 3]) / 2.0
        grasp_center = 1.0 * grasp_center * self.info.crop_image_size / 200 + np.array([self.info.top_left_corner_h, self.info.top_left_corner_w])
        return grasp_center.mean(axis=0).astype(np.int)

    @staticmethod
    def get_label(center_point, angle, width, factor):
        """
        Obtain grasp labels. Each grasp label is represented as two points.
        :param center_point: A numpy array of float32. The position of the object which the gripper will reach.
        :param angle: A float. The rotated angle of the gripper.
        :param width: A float. The width between two finger tips.
        :return: A numpy array of float. The grasp points.
        """
        width = int(width / factor)
        tip_width = int(7 / factor)
        c_h = center_point[0]
        c_w = center_point[1]
        h = [delta_h for delta_h in range(-tip_width/2, tip_width/2)]
        w = [-width / 2, width / 2]
        points = np.asanyarray([[hh, w[0], hh, w[1]] for hh in h])
        rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        points[:, 0:2] = np.dot(rotate_matrix, points[:, 0:2].T).T
        points[:, 2:] = np.dot(rotate_matrix, points[:, 2:].T).T
        points = points + np.asanyarray([[c_h, c_w, c_h, c_w]])
        points = np.floor(points).astype(np.int)
        return points

    @staticmethod
    def get_grasp_angle(grasp_label):
        """
        Get grasp angle.
        :param grasp_label: A numpy array with shape [1, 4] which represents
        a grasp label formulated as 2 points (i.e., (x1, y1, x2, y2)).
        :return: angle_indices: A list of int with length 2. The discretized angles ranged from 0 to 15.
        Besides the original grasp angle, the list contains another angle flipped vertically by original one.
        """
        pt1 = grasp_label[0:2]
        pt2 = grasp_label[2:]
        angle = np.arctan2(pt2[0] - pt1[0], pt2[1] - pt1[1])
        return -angle

    @staticmethod
    def get_gripper_width(grasp_label):
        pt1 = grasp_label[0:2]
        pt2 = grasp_label[2:]
        gripper_width = np.sqrt(np.sum(np.power(pt1-pt2, 2)))
        return gripper_width

    @staticmethod
    def draw_label(points, width, height, color=(255, 0, 0)):
        """
        Draw labels according to the grasp points.
        :param points: A numpy array of float. The grasp points.
        :param width: A float. The width of image.
        :param height: A float. The height of image.
        :param color: A tuple. The color of lines.
        :return: A numpy array of uint8. The label map.
        """
        label = np.ones((height, width, 3), dtype=np.uint8) * 255
        for point in points:
            pt1 = (point[1], point[0])
            pt2 = (point[3], point[2])
            cv2.line(label, pt1, pt2, color)
        return label
