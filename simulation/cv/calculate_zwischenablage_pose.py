from cv.camera import Camera
from cv.block_localization import *
from utils.utils import get_cam_params_from_matrix


def get_zwischenablage_pose(images, cam_mtx, cam_dist):
    tag_id = 356
    detector = dt_apriltags.Detector(nthreads=detection_threads, quad_decimate=quad_decimate)

    cam_params = get_cam_params_from_matrix(cam_mtx)

    corrections = {tag_id: {'pos_corr': [0, 0, 0], 'quat_corr': [1, 0, 0, 0]}}

    quats = []
    positions = []
    for i, im in enumerate(images):
        print(f"Image #{i}")
        poses = get_tag_poses_from_image(im, [tag_id], 48., 255, 56.2, np.array([0, 0, 0]), cam_params, corrections,
                                          detector, cam_mtx, cam_dist)

        pose_mtx = poses[tag_id]
        quat = Quaternion(matrix=pose_mtx)
        quat = Quaternion(quat[0], quat[2], quat[1], -quat[3])
        pos = pose_mtx[:3, 3]
        x = pos[1]
        y = pos[0]
        z = -pos[2]
        pos = np.array([x, y, z])
        quats.append(quat)
        positions.append(pos)

    # average quaternions
    quat = average_quaternions(quats)

    # average positions
    positions = np.reshape(positions, (len(positions), 3))
    pos = np.mean(positions, axis=0)

    transform_mtx = quat.transformation_matrix @ get_Rz_h(90, 'degrees')
    transform_mtx[:3, 3] = pos
    pose = matrix2pose_ZYX(transform_mtx)
    pose[3:] = pose[3:] / math.pi * 180
    pose[3:] = pose[:-4:-1]

    return pos, quat, transform_mtx, pose