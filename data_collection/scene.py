from data_collection.ur import UR5
from data_collection.rg2 import RG2
import numpy as np
import data_collection.vrep as vrep
import time
import cv2


class Scene(object):
    def __init__(self, ip='127.0.0.1',
                 port=19997,
                 camera_handle='realsense'):
        """
        Initialization.Connect to remote server, initialize ur5 and gripper.
        :param ip: The ip address where the server is located.
        :param port: The port number where to connected.
        :param camera_handle: The camera handle id.
        """
        vrep.simxFinish(-1)
        self.client_id = vrep.simxStart(ip, port, True, True, 5000, 5)
        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_blocking)
        self.ur5 = UR5(self.client_id)
        self.gripper = RG2(self.client_id)
        self.ur5.initialization()
        _, self.camera_handle = vrep.simxGetObjectHandle(self.client_id, camera_handle, vrep.simx_opmode_blocking)
        self.camera_height = self.get_object_position(self.camera_handle)[-1]

        self.background_color = self.get_color_image()
        self.background_depth = self.get_depth_image()

    def get_object_handle(self, handle_name):
        """
        Get object handle.
        :param handle_name: A string. Handle name with respect to specific object handle in vrep.
        :return: The number of object handle.
        """
        _, obj = vrep.simxGetObjectHandle(self.client_id, handle_name, vrep.simx_opmode_blocking)
        return obj

    def set_object_position(self, object_handle, position, relative_to_object_handle=-1):
        """
        Set object position.
        :param object_handle: Handle of the object.
        :param position: The position value.
        :param relative_to_object_handle: Indicates relative to which reference frame the position is specified.
        :return: None.
        """
        vrep.simxSetObjectPosition(self.client_id,
                                   object_handle,
                                   relative_to_object_handle,
                                   position,
                                   vrep.simx_opmode_oneshot)

    def get_object_position(self, object_handle, relative_to_object_handle=-1):
        """
        Get object position.
        :param object_handle: An int. Handle of the object.
        :param relative_to_object_handle: An Int. Indicates relative to which reference frame the position is specified.
        :return: A list of float. The position of object.
        """
        _, _, pos, _, _ = self.ur5.func(functionName='pyGetObjectQPosition',
                                        inputInts=[object_handle, relative_to_object_handle],
                                        inputFloats=[],
                                        inputStrings=[],
                                        inputBuffer='')
        return pos

    def set_object_quaternion(self, object_handle, quaternion, relative_to_object_handle=-1):
        """
        Set object quaternion.
        :param object_handle: An int. Handle of the object.
        :param quaternion: A list of float. The quaternion value (x, y, z, w)
        :param relative_to_object_handle: An Int. Indicates relative to which reference frame the object is specified.
        :return: None.
        """
        vrep.simxSetObjectQuaternion(self.client_id,
                                     object_handle,
                                     relative_to_object_handle,
                                     quaternion,
                                     vrep.simx_opmode_oneshot)

    def get_object_quaternion(self, object_handle, relative_to_object_handle=-1):
        """
        Get object quaternion.
        :param object_handle: An int. Handel of the object.
        :param relative_to_object_handle: An int. Indicates relative to which refer
        :return: A list of float. The quaternion of object.
        """
        _, _, quat, _, _ = self.ur5.func(functionName='pyGetObjectQuaternion',
                                         inputInts=[object_handle, relative_to_object_handle],
                                         inputFloats=[],
                                         inputStrings=[],
                                         inputBuffer='')
        return quat

    def replace_object(self, object_handle, object_quat, interval):
        """
        Replace object if it is out of the workspace.
        :param object_handle: An int. Handel of the object.
        :param object_quat: A list of float. The quaternion which the object should rotate.
        :param interval: A list of float. The interval of position where the object should be placed.
        :return: A boolean. A boolean value indicating whether the object is replaced.
        """
        pos = self.get_object_position(object_handle)
        if not interval[0] <= pos[0] <= interval[1] or not interval[0] <= pos[1] <= interval[1]:
            pos[0] = np.random.uniform(interval[0], interval[1])
            pos[1] = np.random.uniform(interval[0], interval[1])
            self.set_object_position(object_handle, pos)
            self.set_object_quaternion(object_handle, object_quat)
            time.sleep(0.7)
            return True
        return False

    def get_color_image(self):
        """
        Get color image.
        :return: A numpy array of uint8. The RGB image containing the whole workspace.
        """
        res, resolution, color = vrep.simxGetVisionSensorImage(self.client_id,
                                                               self.camera_handle,
                                                               0,
                                                               vrep.simx_opmode_blocking)
        return np.asanyarray(color, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)[::-1, ...]

    def get_depth_image(self):
        """
        Get depth image.
        :return: A numpy array of uint16. The depth image containing the whole workspace.
        """
        res, resolution, depth = vrep.simxGetVisionSensorDepthBuffer(self.client_id,
                                                                     self.camera_handle,
                                                                     vrep.simx_opmode_blocking)
        depth = 1000 * np.asanyarray(depth)
        return depth.astype(np.uint16).reshape(resolution[1], resolution[0])[::-1, ...]

    def get_center_from_image(self, depth):
        """
        Get grasp position which belongs to the object in image space.
        :param depth: A numpy array of uint16. The depth image.
        :return: center_point: A numpy array of float32. The position of the object which the gripper will reach.
        :return: points: A numpy array of float32. The object positions which belong to the object in image space.
        :return: A float. The distance between the camera and the center point of object which the gripper will reach.
        """
        crop_depth = depth[45:435, 119:509]
        crop_b_depth = self.background_depth[45:435, 119:509]
        resized_depth = cv2.resize(crop_depth, (200, 200)).astype(np.float32)
        resized_b_depth = cv2.resize(crop_b_depth, (200, 200)).astype(np.float32)
        diff_depth = np.abs(resized_depth - resized_b_depth)
        cv2.medianBlur(diff_depth, 5)
        points = np.stack(np.where(diff_depth > 0), axis=0).T
        center_point = points[np.random.randint(0, points.shape[0])].tolist()
        reaching_height = max(0.02, diff_depth[center_point[0], center_point[1]]*0.0001 - 0.027)
        return center_point, points, reaching_height

    def stop_simulation(self):
        """
        Stop vrep simulation.
        :return: None.
        """
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)
        vrep.simxFinish(self.client_id)

    @staticmethod
    def get_label(center_point, angle, width, ratio):
        """
        Obtain grasp labels. Each grasp label is represented as two points.
        :param center_point: A numpy array of float32. The position of the object which the gripper will reach.
        :param angle: A float. The rotated angle of the gripper.
        :param width: A float. The width between two finger tips.
        :param ratio: A float. The ratio of the width of workspace and the width of image.
        :return: A numpy array of float. The grasp points.
        """
        p_width = width / ratio
        c_h = center_point[0]
        c_w = center_point[1]
        h = [delta_h for delta_h in range(-3, 4)]
        w = [-p_width / 2, p_width / 2]
        points = np.asanyarray([[hh, w[0], hh, w[1]] for hh in h])
        rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        points[:, 0:2] = np.dot(rotate_matrix, points[:, 0:2].T).T
        points[:, 2:] = np.dot(rotate_matrix, points[:, 2:].T).T
        points = points + np.asanyarray([[c_h, c_w, c_h, c_w]])
        points = np.floor(points).astype(np.int)
        return points

    @staticmethod
    def draw_label(points, width, height, color=(0, 255, 0)):
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
