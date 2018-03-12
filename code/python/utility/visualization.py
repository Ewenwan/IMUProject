import os
import math

import numpy as np
import plyfile
import quaternion
import cv2

def write_ply_to_file(path, position, orientation, acceleration=None,
                      global_rotation=np.identity(3, float), local_axis=None,
                      trajectory_color=None, num_axis=3,
                      length=1.0, kpoints=100, interval=100):
    """
    Visualize camera trajectory as ply file
    :param path: path to save
    :param position: Nx3 array of positions
    :param orientation: Nx4 array or orientation as quaternion
    :param acceleration: (optional) Nx3 array of acceleration
    :param global_rotation: (optional) global rotation
    :param local_axis: (optional) local axis vector
    :param trajectory_color: (optional) the color of the trajectory. The default is [255, 0, 0] (red)
    :return: None
    """
    num_cams = position.shape[0]
    assert orientation.shape[0] == num_cams

    max_acceleration = 1.0
    if acceleration is not None:
        assert acceleration.shape[0] == num_cams
        max_acceleration = max(np.linalg.norm(acceleration, axis=1))
        print('max_acceleration: ', max_acceleration)
        num_axis = 4

    sample_pt = np.arange(0, num_cams, interval, dtype=int)
    num_sample = sample_pt.shape[0]

    # Define the optional transformation. Default is set w.r.t tango coordinate system
    position_transformed = np.matmul(global_rotation, np.array(position).transpose()).transpose()
    if local_axis is None:
        local_axis = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0]])
    if trajectory_color is None:
        trajectory_color = [0, 255, 255]
    axis_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]
    vertex_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    positions_data = np.empty((position_transformed.shape[0],), dtype=vertex_type)
    positions_data[:] = [tuple([*i, *trajectory_color]) for i in position_transformed]

    app_vertex = np.empty([num_axis * kpoints], dtype=vertex_type)
    for i in range(num_sample):
        q = quaternion.quaternion(*orientation[sample_pt[i]])
        if acceleration is not None:
            local_axis[:, -1] = acceleration[sample_pt[i]].flatten() / max_acceleration
        global_axes = np.matmul(global_rotation, np.matmul(quaternion.as_rotation_matrix(q), local_axis))
        for k in range(num_axis):
            for j in range(kpoints):
                axes_pts = position_transformed[sample_pt[i]].flatten() +\
                           global_axes[:, k].flatten() * j * length / kpoints
                app_vertex[k*kpoints + j] = tuple([*axes_pts, *axis_color[k]])

        positions_data = np.concatenate([positions_data, app_vertex], axis=0)
    vertex_element = plyfile.PlyElement.describe(positions_data, 'vertex')
    plyfile.PlyData([vertex_element], text=True).write(path)


def read_trajectory_from_ply_file(path):
    with open(path, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
    return plydata.elements[0].data


def draw_trajectory_on_image(img, traj, color, thickness, rescale=True):
    traj_scaled = traj
    if rescale:
        traj_center = np.sum(traj, axis=0) / traj.shape[0]
        r_max = np.max(np.linalg.norm(traj - traj_center, axis=1))
        scale_factor = min(img.shape[0]/2, img.shape[1]/2) * 0.8 / r_max
        traj_scaled = (traj - traj_center) * scale_factor
        traj_scaled[:, 0] += img.shape[1] / 2
        traj_scaled[:, 1] += img.shape[0] / 2
    traj_scaled = traj_scaled[:, :2].astype(np.int32)
    for i in range(1, traj_scaled.shape[0]):
        cv2.line(img, tuple(traj_scaled[i-1]), tuple(traj_scaled[i]), color, thickness)
    cv2.circle(img, tuple(traj_scaled[0]), 3, (255, 255, 0), 3)
    return img


def get_trajectory_visualization(traj, width, height, color, thickness, rescale=True):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return draw_trajectory_on_image(img, traj, color, thickness, rescale)


def read_map_file(path):
    """
    Read the meta information for the floor plan image.

    Args:
        path: the path to the file.
    Returns:
        img: The image file as a numpy array;
        meter_per_pixel: The scale as meters/pixel.
        origin: The pixel location of the world origin point.
        init_heading: The initial headings.
    """
    with open(path) as f:
        dir_name = os.path.dirname(path)
        img = cv2.imread(dir_name + '/' + f.readline().strip())
        assert img.shape[0] > 0 and img.shape[1] > 0, 'Can not open image file'
        meter_per_pixel = float(f.readline().strip())
        ori_str = f.readline().strip().split()
        origin = np.array([int(ori_str[0]), int(ori_str[1])])
        init_heading = float(ori_str[2])
    return img, meter_per_pixel, origin, init_heading


def rescale_trajectory_legacy(positions, meter_per_pixel, img_origin, img_heading):
    """
    Rescale and rotate the trajectory based on the scale, origin and initial heading.
    This function performs 3D transformation, which is unnecessary. Consider use
    "rescale_trajectory" instead.
    Args:
        positions: Nx3 array representing the trajectory.
        meter_per_pixel: The distance in the real world represented by one pixel.
        img_origin: The pixel location of the world origin point.
        img_heading: The initial heading.
    Returns:
        new_pos: Rescaled and rotated trajectory.
    """
    rotor = quaternion.from_euler_angles(0, 0, img_heading)
    new_pos = np.empty(positions.shape)
    for i in range(new_pos.shape[0]):
        rotated = (rotor * quaternion.quaternion(0., positions[i][0], -positions[i][1], positions[i][2])
                   * rotor.conj()).vec
        new_pos[i] = rotated / meter_per_pixel + img_origin
    return new_pos


def rescale_trajectory(positions, meter_per_pixel, img_origin, img_heading):
    """
    Rescale and rotate the trajectory based on the scale, origin and initial heading.
    """
    rotor = np.array([[math.cos(img_heading), -math.sin(img_heading)],
                     [math.sin(img_heading), math.cos(img_heading)]])
    new_pos = np.copy(positions)
    # The Y axis is inverted
    new_pos[:, 1] *= -1
    new_pos[:, :2] = (np.matmul(rotor, new_pos[:, :2].T) / meter_per_pixel + img_origin[:2, None]).T
    return new_pos


def map_overlay(img, positions, meter_per_pixel, img_origin, img_heading, color=(255, 0, 0), thickness=2):
    """
    Draw the trajectory on a floor-plan image.

    Args:
        img: The input floor-plan image. This image will not be changed.
        positions, meter_per_pixel, img_origin, img_heading: The same as "rescale_trajectory" function.
        color: The color of the trajectory. Note that it's in BGR order.
        thickness: The thickness of the trajectory.
    Returns:
        img_out: The floor-plan image with the trajectory overlaid on.
    """
    img_out = np.copy(img)
    new_pos = rescale_trajectory(positions, meter_per_pixel, img_origin, img_heading)
    for pos in new_pos:
        cv2.circle(img_out, (int(pos[0]), int(pos[1])), thickness, color, thickness)
    return img_out

