from functools import partial, reduce
from math import sqrt
import data_collection.vrep as vrep
import time


class RG2(object):
    def __init__(self, client_id, script_name='UR5', motor_vel=0.11, motor_force=77):
        """
        Initialization for RG2.
        :param client_id: An int. The client ID. refer to simxStart.
        :param script_name: A string. The lua script name in vrep.
        :param motor_vel: A float. Target velocity of the joint1.
        :param motor_force: The maximum force or torque that the joint can exert.
        """
        self.client_id = client_id
        self.motor_vel = motor_vel
        self.motor_force = motor_force
        _, self.attach_point = vrep.simxGetObjectHandle(self.client_id,
                                                        'RG2_attachPoint',
                                                        vrep.simx_opmode_blocking)
        _, self.proximity_sensor = vrep.simxGetObjectHandle(self.client_id,
                                                            'RG2_attachProxSensor',
                                                            vrep.simx_opmode_blocking)
        _, self.right_touch = vrep.simxGetObjectHandle(self.client_id,
                                                       'RG2_rightTouch',
                                                       vrep.simx_opmode_blocking)
        _, self.left_touch = vrep.simxGetObjectHandle(self.client_id,
                                                      'RG2_leftTouch',
                                                      vrep.simx_opmode_blocking)
        _, self.joint = vrep.simxGetObjectHandle(self.client_id,
                                                 'RG2_openCloseJoint',
                                                 vrep.simx_opmode_blocking)
        self.func = partial(vrep.simxCallScriptFunction,
                            clientID=self.client_id,
                            scriptDescription=script_name,
                            options=vrep.sim_scripttype_childscript,
                            operationMode=vrep.simx_opmode_blocking)

    def open_gripper(self):
        """
        Open gripper.
        :return: None.
        """
        vrep.simxSetJointTargetVelocity(self.client_id,
                                        self.joint,
                                        self.motor_vel,
                                        vrep.simx_opmode_streaming)
        vrep.simxSetJointForce(self.client_id,
                               self.joint,
                               self.motor_force,
                               vrep.simx_opmode_oneshot)
        self.wait_until_stop(self.right_touch)

    def close_gripper(self):
        """
        Close gripper.
        :return: None
        """
        vrep.simxSetJointTargetVelocity(self.client_id,
                                        self.joint,
                                        -self.motor_vel,
                                        vrep.simx_opmode_oneshot)
        vrep.simxSetJointForce(self.client_id,
                               self.joint,
                               self.motor_force,
                               vrep.simx_opmode_oneshot)
        self.wait_until_stop(self.right_touch)

    def attach_object(self, object_handle):
        """
        Attach object to the gripper. This is an alternative to grasp objects.
        :param object_handle: An int. The handle of object which is successfully grasped by gripper.
        :return: None
        """
        vrep.simxSetObjectParent(self.client_id,
                                 object_handle,
                                 self.attach_point,
                                 True,
                                 vrep.simx_opmode_blocking)

    def untie_object(self, object_handle):
        """
        Untie object to the gripper. This is an alternative to grasp objects.
        :param object_handle: An int. The handle of object which is attached to the gripper.
        :return: None
        """
        vrep.simxSetObjectParent(self.client_id,
                                 object_handle,
                                 -1,
                                 True,
                                 vrep.simx_opmode_blocking)

    def get_object_detection_status(self, threshold=0.005):
        """
        Detect whether gripper grasp object successfully.
        :param threshold: A float. When distance between two gripper tips is larger than threshold,
        gripper successfully grasp object, vice versa.
        :return: A boolean. Return object detection status.
        """
        time.sleep(0.7)
        _, d_pos = vrep.simxGetObjectPosition(self.client_id,
                                              self.left_touch,
                                              self.right_touch,
                                              vrep.simx_opmode_blocking)
        if threshold < abs(d_pos[0]) < 0.085:
            half_touch = 0.01475 - 0.005
            # distance from proximity sensor to left touch.
            _, d_p2t_l = vrep.simxGetObjectPosition(self.client_id,
                                                    self.left_touch,
                                                    self.proximity_sensor,
                                                    vrep.simx_opmode_blocking)
            # distance from proximity sensor to right touch.
            _, d_p2t_r = vrep.simxGetObjectPosition(self.client_id,
                                                    self.right_touch,
                                                    self.proximity_sensor,
                                                    vrep.simx_opmode_blocking)
            _, distance = self.read_proximity_sensor()
            if distance < d_p2t_l[-1] + half_touch and distance < d_p2t_r[-1] + half_touch:
                _, d_l2r = vrep.simxGetObjectPosition(self.client_id,
                                                      self.left_touch,
                                                      self.right_touch,
                                                      vrep.simx_opmode_blocking)
                diff = abs(d_l2r[-1])
                print(diff)
                if diff < 0.001:
                    return True
        return False

    def get_gripper_width(self):
        """
        Get gripper width.
        :return: Gripper width.
        """
        _, d_pos = vrep.simxGetObjectPosition(self.client_id,
                                              self.left_touch,
                                              self.right_touch,
                                              vrep.simx_opmode_blocking)
        return abs(d_pos[0])

    def read_proximity_sensor(self):
        """
        Read proximity sensor.
        :return:
        """
        _, out_ints, out_floats, _, _ = self.func(functionName='pyReadProxSensor',
                                                  inputInts=[],
                                                  inputFloats=[],
                                                  inputStrings=[],
                                                  inputBuffer='')
        return out_ints[0], out_floats[0]

    def wait_until_stop(self, handle, threshold=0.005, time_delay=0.2):
        """
        Wait until the operation finishes.
        This is a delay function called in order to make sure that
        the operation executed has been completed.
        :param handle: An int.Handle of the object.
        :param threshold: A float. The object position threshold.
        If the object positions difference between two time steps is smaller than the threshold,
        the execution completes, otherwise the loop continues.
        :param time_delay:A float. How much time we should wait in an execution step.
        :return: None
        """
        while True:
            _, pos1 = vrep.simxGetObjectPosition(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            _, quat1 = vrep.simxGetObjectQuaternion(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            time.sleep(time_delay)
            _, pos2 = vrep.simxGetObjectPosition(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            _, quat2 = vrep.simxGetObjectQuaternion(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            pose1 = pos1 + quat1
            pose2 = pos2 + quat2
            theta = 0.5 * sqrt(reduce(lambda x, y: x + y, map(lambda x, y: (x - y) ** 2, pose1, pose2)))
            if theta < threshold:
                return
