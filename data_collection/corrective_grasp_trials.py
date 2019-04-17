from math import pi
from scipy import misc
from data_collection.scene import Scene
from functools import partial
import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Data collection in vrep simulator.')
parser.add_argument('--ip',
                    default='127.0.0.1',
                    type=str,
                    help='ip address to the vrep simulator.')
parser.add_argument('--port',
                    default=19997,
                    type=str,
                    help='port to the vrep simulator.')
parser.add_argument('--obj_id',
                    default='obj0',
                    type=str,
                    help='object name in vrep.')
parser.add_argument('--num_grasp',
                    default=200,
                    type=int,
                    help='the number of grasp trails.')
parser.add_argument('--num_repeat',
                    default=1,
                    type=int,
                    help='the number of repeat time if the gripper successfully grasp an object.')
parser.add_argument('--output',
                    default='data/3dnet',
                    type=str,
                    help='directory to save data.')
args = parser.parse_args()

step = pi / 180.0
ratio = 0.4 / 200
interval = [0.23, 0.57]
rand = partial(np.random.uniform, interval[0], interval[1])


def main():
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    data_id = len(os.listdir(args.output))
    data_path = os.path.join(args.output, '{:04d}'.format(data_id))
    color_path = os.path.join(data_path, 'color')
    depth_path = os.path.join(data_path, 'depth')
    height_map_color_path = os.path.join(data_path, 'height_map_color')
    height_map_depth_path = os.path.join(data_path, 'height_map_depth')
    label_path = os.path.join(data_path, 'label')
    os.mkdir(data_path)
    os.mkdir(color_path)
    os.mkdir(depth_path)
    os.mkdir(height_map_color_path)
    os.mkdir(height_map_depth_path)
    os.mkdir(label_path)

    s = Scene(args.ip, args.port, 'realsense')
    init_pos = [45*step, 10*step, 90*step, -10*step, -90*step, -45*step]
    top_pos = [-45*step, 10*step, 90*step, -10*step, -90*step, -45*step]
    # object handle
    obj = s.get_object_handle(args.obj_id)
    obj_quat = s.get_object_quaternion(obj)
    s.gripper.open_gripper()

    misc.imsave(data_path + '/background_color.png', s.background_color)
    cv2.imwrite(data_path + '/background_depth.png', s.background_depth)
    misc.imsave(data_path+'/crop_background_color.png',
                cv2.resize(s.background_color[45:435, 119:509, :], (200, 200)))
    cv2.imwrite(data_path+'/crop_background_depth.png',
                cv2.resize(s.background_depth[45:435, 119:509], (200, 200)))

    f = open(os.path.join(data_path, 'file_name.txt'), 'w')

    s.set_object_position(obj, [rand(), rand(), s.get_object_position(obj)[-1]])

    for i in range(args.num_grasp):  # number of grasp trials.
        print(i)
        s.gripper.open_gripper()
        s.ur5.move_to_joint_position(init_pos)
        s.replace_object(obj, obj_quat, interval)
        color = s.get_color_image()
        depth = s.get_depth_image()
        center_point, object_points, reaching_height = s.get_center_from_image(depth)
        s.ur5.move_to_joint_position(top_pos)
        episode = []
        # move to the upward side with respect to the object
        init_quat = s.ur5.get_end_effector_quaternion()
        target_pos = s.get_object_position(obj)
        target_pos[0] = 0.200 + center_point[0] * ratio
        target_pos[1] = 0.200 + center_point[1] * ratio
        target_pos[2] = reaching_height
        target_pos[-1] += 0.1563
        episode.append(target_pos + init_quat)
        s.ur5.move_to_object_position(target_pos + init_quat)
        # randomly rotate gripper's angle
        joint_angles = s.ur5.get_joint_positions()
        curr_angle = joint_angles[-1] / np.pi * 180.0
        angle = np.random.randint(0, 180)
        joint_angles[-1] = (curr_angle + angle) * step
        s.ur5.move_to_joint_position(joint_angles)
        quat = s.ur5.get_end_effector_quaternion()
        episode.append(target_pos + quat)
        # move to the object downward
        target_pos[-1] -= 0.1563
        episode.append(target_pos + quat)
        s.ur5.move_to_object_position(target_pos + quat)
        # try to grasp object
        s.gripper.close_gripper()
        # determine whether the object is successfully grasped.
        if s.gripper.get_object_detection_status():
            print('grasp success.')
            s.gripper.attach_object(obj)
            width = s.gripper.get_gripper_width()
            s.gripper.open_gripper()
            s.gripper.untie_object(obj)
            s.ur5.move_to_object_position(episode[1])
            s.ur5.move_to_joint_position(init_pos)
            # if we successfully grasp the object,
            # we can record the relative grasp configuration and successively collect the positive samples.
            for j in range(args.num_repeat):  # the number of positive samples we want to collect in this grasp trials.
                if s.replace_object(obj, obj_quat, interval):
                    break
                # rerecord the pattern.
                color = s.get_color_image()
                depth = s.get_depth_image()
                _, object_points, reaching_height = s.get_center_from_image(depth)
                crop_color = cv2.resize(color[45:435, 119:509, :], (200, 200))
                crop_depth = cv2.resize(depth[45:435, 119:509], (200, 200))
                points = s.get_label(center_point, -angle*np.pi/180.0, width, ratio)
                label = s.draw_label(points, 200, 200)
                s.ur5.move_to_joint_position(top_pos)
                s.ur5.move_to_object_position(episode[0])
                s.ur5.move_to_object_position(episode[1])
                s.ur5.move_to_object_position(episode[2])
                s.gripper.close_gripper()
                if s.gripper.get_object_detection_status():
                    s.gripper.attach_object(obj)
                    s.ur5.move_to_object_position(episode[1])
                    # quat = s.ur5.get_end_effector_quaternion()
                    pos = s.ur5.get_end_effector_position()
                    pos[0], pos[1] = rand(), rand()
                    center_point = [int((pos[0] - 0.200) / ratio), int((pos[1] - 0.200) / ratio)]
                    episode[0] = pos + init_quat
                    s.ur5.move_to_object_position(pos + init_quat)
                    joint_angles = s.ur5.get_joint_positions()
                    curr_angle = joint_angles[-1] / np.pi * 180.0
                    angle = np.random.randint(0, 180)
                    joint_angles[-1] = (curr_angle + angle) * step
                    s.ur5.move_to_joint_position(joint_angles)
                    quat = s.ur5.get_end_effector_quaternion()
                    pos = s.ur5.get_end_effector_position()
                    episode[1] = pos + quat
                    pos[-1] = episode[2][2]
                    episode[2] = pos + quat
                    s.ur5.move_to_object_position(pos + quat)
                    s.gripper.open_gripper()
                    s.gripper.untie_object(obj)
                    s.ur5.move_to_object_position(episode[1])
                    s.ur5.move_to_joint_position(init_pos)
                    misc.imsave(color_path + '/{:06d}_{:04d}.png'.format(i, j), color)
                    cv2.imwrite(depth_path + '/{:06d}_{:04d}.png'.format(i, j), depth)
                    misc.imsave(height_map_color_path + '/{:06d}_{:04d}.png'.format(i, j), crop_color)
                    cv2.imwrite(height_map_depth_path + '/{:06d}_{:04d}.png'.format(i, j), crop_depth)
                    misc.imsave(label_path + '/{:06d}_{:04d}.png'.format(i, j), label)
                    np.savetxt(label_path + '/{:06d}_{:04d}.good.txt'.format(i, j), points)
                    np.savetxt(label_path + '/{:06d}_{:04d}.object_points.txt'.format(i, j), np.asanyarray(object_points))
                    f.write('{:06d}_{:04d}\n'.format(i, j))
                else:
                    break

        else:
            print('grasp failed')
            s.gripper.open_gripper()
            s.ur5.move_to_object_position(episode[1])
            crop_color = cv2.resize(color[45:435, 119:509, :], (200, 200))
            crop_depth = cv2.resize(depth[45:435, 119:509], (200, 200))
            points = s.get_label(center_point, -angle * np.pi / 180.0, 0.085, ratio)
            label = s.draw_label(points, 200, 200, (255, 0, 0))
            misc.imsave(color_path + '/{:06d}.png'.format(i), color)
            cv2.imwrite(depth_path + '/{:06d}.png'.format(i), depth)
            misc.imsave(height_map_color_path + '/{:06d}.png'.format(i), crop_color)
            cv2.imwrite(height_map_depth_path + '/{:06d}.png'.format(i), crop_depth)
            misc.imsave(label_path + '/{:06d}.png'.format(i), label)
            np.savetxt(label_path + '/{:06d}.bad.txt'.format(i), points)
            np.savetxt(label_path + '/{:06d}.object_points.txt'.format(i), np.asanyarray(object_points))
            f.write('{:06d}\n'.format(i))

    s.stop_simulation()
    f.close()


if __name__ == '__main__':
    main()
