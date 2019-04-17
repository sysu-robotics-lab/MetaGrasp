import numpy as np


class DataProcessor(object):
    def __init__(self):
        pass

    @staticmethod
    def get_grasp_center(grasp_labels):
        """
        Get grasp center.
        :param grasp_labels: A numpy array with shape [num_labels, 4].
        Each row represents a grasp label formulated as 2 points (i.e., (x1, y1, x2, y2)).
        :return: grasp_center: A numpy array with shape [num_labels, 2].
        Each row represents a grasp center corresponding to the grasp label.
        """
        row, _ = grasp_labels.shape
        grasp_center = np.empty((row, 2), dtype=np.float32)
        grasp_center[:, 0] = (grasp_labels[:, 0] + grasp_labels[:, 2]) / 2.0
        grasp_center[:, 1] = (grasp_labels[:, 1] + grasp_labels[:, 3]) / 2.0
        return grasp_center

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
        if angle < 0:
            angle += np.pi * 2
        angle_indices = []
        angle_indices.append(int(round(angle / ((22.5 / 360.0) * np.pi * 2))))
        if angle >= np.pi:
            angle_indices.append(int(round((angle - np.pi) / ((22.5 / 360.0) * np.pi * 2))))
        else:
            angle_indices.append(int(round((angle + np.pi) / ((22.5 / 360.0) * np.pi * 2))))
        return angle_indices

    @staticmethod
    def rotate(points, center, angle):
        """
        Rotate points.
        :param points: A numpy array with shape [num_points, 2].
        :param center: A numpy array with shape [1, 2]. The rotation center of the points.
        :param angle: A float. The rotated angle represented in radian.
        :return: points: A numpy array with shape [num_points, 2]. The rotated points.
        """
        points = points.copy()
        h = center[0]
        w = center[1]
        points[:, 0] -= h
        points[:, 1] -= w
        rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        points = np.dot(rotate_matrix, points.T).T
        points[:, 0] += h
        points[:, 1] += w
        return points

    @staticmethod
    def get_diff_depth(depth_o, depth_b):
        """
        Get difference depth image by subtracting current depth image and background depth image.
        :param depth_o: A numpy array with shape [height, width]. The current depth image.
        :param depth_b: A numpy array with shape [height, width]. The background depth image.
        :return: diff_depth: A numpy array with shape [height, width]. The difference depth image.
        """
        diff_depth = depth_b - depth_o
        diff_depth[np.where(diff_depth < 0)] = 0
        # diff_depth = cv2.medianBlur(diff_depth, 3)
        diff_depth = diff_depth.astype(np.uint16)
        return diff_depth

    @staticmethod
    def encode_depth(depth):
        """
        Encode depth image to RGB format.
        :param depth: A numpy array with shape [height, width]. The depth image.
        :return:
        """
        r = depth / 256 / 256
        g = depth / 256
        b = depth % 256
        # encoded_depth = np.stack([r, g, b], axis=2).astype(np.uint8)
        encoded_depth = np.stack([b, g, r], axis=2).astype(np.uint8)  # use bgr order due to cv2 format
        return encoded_depth

    @staticmethod
    def gaussianize_label(grasp_label, grasp_centers, camera_height, is_good, expand=True):
        if is_good:
            for grasp_center in grasp_centers:
                if grasp_label[grasp_center[0], grasp_center[1], 0] == 255:
                    # grasp_label[grasp_center[0], grasp_center[1], 0] = 0
                    # grasp_label[grasp_center[0], grasp_center[1], 1] = 255
                    pass
                else:
                    left = right = 0
                    while grasp_label[grasp_center[0], grasp_center[1]-left, 0] == 0:
                        left += 1
                    while grasp_label[grasp_center[0], grasp_center[1]+right, 0] == 0:
                        right += 1

                    def gauss(x, c, sigma):
                        return int(255 * np.exp(-(x-c) ** 2 / (2 * sigma ** 2)))
                    width = left + right
                    # width = min(left, right) * 2
                    grasp_center[1] += width / 2 - left
                    # sigma = camera_height * width
                    # sigma = width / (camera_height * 10.0)
                    sigma = 0.03 / (0.4 * camera_height / 0.57) * 200.0 / 3.0
                    for idx in range(grasp_center[1]-width/2, grasp_center[1]+width/2+1):
                        value = gauss(idx, grasp_center[1], sigma)
                        grasp_label[grasp_center[0], idx, 1] = value
                        grasp_label[grasp_center[0], idx, 2] = 255 - value
        else:
            if expand:
                for grasp_center in grasp_centers:
                    if grasp_label[grasp_center[0], grasp_center[1], 0] == 255:
                        grasp_label[grasp_center[0], grasp_center[1], 0] = 0
                        grasp_label[grasp_center[0], grasp_center[1], 2] = 255
                    else:
                        left = right = 0
                        while grasp_label[grasp_center[0], grasp_center[1]-left, 0] == 0:
                            grasp_label[grasp_center[0], grasp_center[1]-left, 2] = 255
                            left += 1
                        while grasp_label[grasp_center[0], grasp_center[1]+right, 0] == 0:
                            grasp_label[grasp_center[0], grasp_center[1]+right, 2] = 255
                            right += 1
            else:
                grasp_label[grasp_centers[:, 0], grasp_centers[:, 1], 0] = 0
                grasp_label[grasp_centers[:, 0], grasp_centers[:, 1], 2] = 255
        return grasp_label



