from functools import partial, reduce
from math import sqrt
import time

import data_collection.vrep as vrep


class UR5(object):
    def __init__(self, client_id):
        """
        Initialization.
        :param client_id: An int. The client ID. refer to simxStart.
        """
        self.client_id = client_id
        self.joint_handles = [vrep.simxGetObjectHandle(self.client_id,
                                                       'UR5_joint{}'.format(i),
                                                       vrep.simx_opmode_blocking)[1] for i in range(1, 6+1)]
        _, self.ik_tip_handle = vrep.simxGetObjectHandle(self.client_id,
                                                         'UR5_ik_tip',
                                                         vrep.simx_opmode_blocking)
        _, self.ik_target_handle = vrep.simxGetObjectHandle(self.client_id,
                                                            'UR5_ik_target',
                                                            vrep.simx_opmode_blocking)
        self.func = partial(vrep.simxCallScriptFunction,
                            clientID=self.client_id,
                            scriptDescription="UR5",
                            options=vrep.sim_scripttype_childscript,
                            operationMode=vrep.simx_opmode_blocking)
        self.initialization()

    def initialization(self):
        """
        Call script function pyInit in vrep child script.
        :return: None
        """
        _ = self.func(functionName='pyInit',
                      inputInts=[],
                      inputFloats=[],
                      inputStrings=[],
                      inputBuffer='')
        time.sleep(0.7)

    def wait_until_stop(self, handle, threshold=0.01):
        """
        Wait until the operation finishes.
        This is a delay function called in order to make sure that
        the operation executed has been completed.
        :param handle: An int.Handle of the object.
        :param threshold: A float. The object position threshold.
        If the object positions difference between two time steps is smaller than the threshold,
        the execution completes, otherwise the loop continues.
        :return: None
        """
        while True:
            _, pos1 = vrep.simxGetObjectPosition(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            _, quat1 = vrep.simxGetObjectQuaternion(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            time.sleep(0.7)
            _, pos2 = vrep.simxGetObjectPosition(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            _, quat2 = vrep.simxGetObjectQuaternion(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            pose1 = pos1 + quat1
            pose2 = pos2 + quat2
            theta = 0.5 * sqrt(reduce(lambda x, y: x + y, map(lambda x, y: (x - y) ** 2, pose1, pose2)))
            if theta < threshold:
                return

    def enable_ik(self, enable=0):
        """
        Call script function pyEnableIk in vrep child script.
        :param enable: A int. Whether to enable inverse kinematic.
        If enable = 1 the ur5 enables ik, and vice versa.
        :return: None
        """
        _ = self.func(functionName='pyEnableIk',
                      inputInts=[enable],
                      inputFloats=[],
                      inputStrings=[],
                      inputBuffer='')

    def move_to_joint_position(self, joint_angles):
        """
        Moves (actuates) several joints at the same time using the Reflexxes Motion Library type II or IV.
        Call script function pyMoveToJointPositions in vrep child script.
        :param joint_angles: A list of floats. The desired target angle positions of the joints.
        :return: None
        """
        self.enable_ik(0)
        _ = self.func(functionName='pyMoveToJointPositions',
                      inputInts=[],
                      inputFloats=joint_angles,
                      inputStrings=[],
                      inputBuffer='')
        self.wait_until_stop(self.ik_target_handle)
        time.sleep(0.7)  # todo: remove manual time delay

    def move_to_object_position(self, pose):
        """
        Moves an object to a given position and/or orientation using Reflexxes Motion Library type II or IV.
        Call script function pyMoveToPosition in vrep child script.
        :param pose: A list of floats. The desired target position of the object.
        :return: None
        """
        self.enable_ik(1)
        _ = self.func(functionName='pyMoveToPosition',
                      inputInts=[],
                      inputFloats=pose,
                      inputStrings=[],
                      inputBuffer='')
        self.wait_until_stop(self.ik_target_handle)

    def get_end_effector_position(self):
        """
        Get end effector position.
        :return: pos: A list of floats. The position of end effector.
        """
        _, pos = vrep.simxGetObjectPosition(self.client_id,
                                            self.ik_tip_handle,
                                            -1,
                                            vrep.simx_opmode_blocking)
        return pos

    def get_end_effector_quaternion(self):
        """
        Get end effector quaternion.
        :return: quat: A list of floats. The angle of end effector represented by quaternion.
        """
        # _, quat = vrep.simxGetObjectQuaternion(self.client_id,
        #                                        self.ik_tip_handle,
        #                                        -1,
        #                                        vrep.simx_opmode_blocking)
        _, _, quat, _, _ = self.func(functionName='pyGetObjectQuaternion',
                                     inputInts=[self.ik_tip_handle, -1],
                                     inputFloats=[],
                                     inputStrings=[],
                                     inputBuffer='')
        return quat

    def get_joint_positions(self, is_first_call=False):
        """
        Get joint angle positions.
        :param is_first_call: A boolean. Specify which operation mode vrep api function chooses.
        :return: joint_positions: A list of float. Return value of UR joint angles.
        """
        if is_first_call:
            opmode = vrep.simx_opmode_streaming
        else:
            opmode = vrep.simx_opmode_blocking
        joint_positions = [vrep.simxGetJointPosition(self.client_id,
                                                     joint_handle,
                                                     opmode)[1] for joint_handle in self.joint_handles]
        return joint_positions


