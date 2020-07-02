from utils.utils import translation_along_axis_towards_point
from constants import *
import cv2
from cv.camera import Camera
from cv.block_localization import get_tag_poses_from_image
import dt_apriltags
from cv.transformations import matrix2pose
import json
from os import path
import csv

images_num = 3
target_tag_size = 9.6
ref_tag_id = 224
ref_tag_size = 40
ref_tag_pos = np.array([0, 0, 0])

"""Translation defined along axes of the tag itself (not global axes)"""
def compute_translational_correction(desired_pos, actual_pos, corrected_quat):
    x_corr = translation_along_axis_towards_point(actual_pos, desired_pos, x_unit_vector, corrected_quat)
    y_corr = translation_along_axis_towards_point(actual_pos, desired_pos, y_unit_vector, corrected_quat)
    z_corr = translation_along_axis_towards_point(actual_pos, desired_pos, z_unit_vector, corrected_quat)

    return np.array([x_corr, y_corr, z_corr])

"""q_reference = q_actual * q_correction
=> q_corr = inv(q_actual) * q_ref"""
def compute_corrective_quat(desired_quat, actual_quat):
    desired_quat = Quaternion(desired_quat)
    actual_quat = Quaternion(actual_quat)
    return actual_quat.inverse * desired_quat

def correct_position(current_pos, correction, quat):
    res = current_pos
    res += correction[0] * quat.rotate(x_unit_vector)
    res += correction[1] * quat.rotate(y_unit_vector)
    res += correction[2] * quat.rotate(z_unit_vector)

    return res

def correct_quat(current_quat, corrective_quat):
    return current_quat * corrective_quat

def show_message_and_wait(text):
    # Create a black image
    w = 500
    h = 300
    img = np.zeros((h, w, 3), np.uint8)

    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, h//2)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    # Display the image
    cv2.imshow("img", img)
    cv2.waitKey(0)

def show_image_with_number(im, n):
    h, w = im.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 20
    fontColor = (0, 0, 255)
    lineType = 15
    text = str(n)

    cv2.putText(im, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    # Display the image
    cv2.imshow('img', im)
    key = cv2.waitKey(0)
    print(f"Key: {key}")

def take_images_and_return_corrections(cam, camera_params, detector):
    measured_positions = []
    measured_quats = []
    show_message_and_wait('Put the block on position')
    for i in range(images_num):
        im = cam.take_picture()
        show_image_with_number(im, i+1)
        target_tag_ids = [i for i in range(112)]
        mtx = get_tag_poses_from_image(im, target_tag_ids, target_tag_size, ref_tag_id,
                                       ref_tag_size, ref_tag_pos, camera_params, detector)

        print(f"mtx = {mtx}")
        while mtx is None or len(mtx) != 1:
            show_message_and_wait("Tag not detected!")
            im = cam.take_picture()
            show_image_with_number(im, i + 1)
            mtx = get_tag_poses_from_image(im, target_tag_ids, target_tag_size, ref_tag_id,
                                           ref_tag_size, ref_tag_pos, camera_params, detector)

        for id in mtx:
            if id != ref_tag_id:
                target_tag_id = id

        measured_positions.append(matrix2pose(mtx[target_tag_id])[:3])
        measured_quats.append(Quaternion(matrix=mtx[target_tag_id]))

    # average measured positions and quaternions
    mean_pos = np.mean(measured_positions, axis=0)
    mean_quat = np.mean(np.array([q.q for q in measured_quats]), axis=0)
    pos_std = np.std(measured_positions, axis=0)
    quat_std = np.std(np.array([q.q for q in measured_quats]), axis=0)

    return mean_pos, mean_quat, target_tag_id, pos_std, quat_std

def collect_poses(cam, camera_params, detector, fname, n):
    for i in range(n):
        measured_pos, measured_quat, tag_id, pos_std, quat_std = take_images_and_return_corrections(cam, camera_params, detector)
        write_data_to_file(fname,
                           tag_id,
                           list(measured_pos),
                           list(measured_quat),
                           list(pos_std),
                           list(quat_std))

def write_data_to_file(filename, tag_id, measured_pos, measured_quat, pos_std, quat_std):
    # this function should preserve data stored in file
    # for this purpose we first read all data from the file

    # create file if not exists
    if not path.exists(filename):
        with open(filename, mode='w') as f:
            d = dict()
            json.dump(d, f)

    with open(filename, mode='r') as f:
        data = json.load(f)

    # add new data row to the data
    data[str(tag_id)] = {'pos': measured_pos, 'quat': measured_quat, 'pos_std': pos_std, 'quat_std': quat_std}

    print(data)

    # write data back to the file
    with open(filename, mode='w') as f:
        json.dump(data, f)
    print(f"Data for the tag #{tag_id} successfully writen!")

def calculate_reference_pos(block_width, block_height):
    ref_tag_plane_width = 50
    ref_tag_plane_height = 50
    wall_thickness = 5
    x = ref_tag_plane_width / 2 + block_width / 2
    y = ref_tag_plane_height/2 - wall_thickness - block_height/2
    z = 0

    return np.array([x, y, z])

def create_file_with_corrections(block_sizes, measured_poses, filename):
    corrections = dict()
    get_block_id = lambda x: x//2
    for id in measured_poses:
        height = float(block_sizes[str(get_block_id(int(id)))]['height'])
        width = float(block_sizes[str(get_block_id(int(id)))]['width'])
        reference_pos = calculate_reference_pos(width, height)
        measured_pos = measured_poses[id]['pos']
        measured_quat = measured_poses[id]['quat']
        ref_quat = Quaternion([1, 0, 0, 0])
        corrective_quat = compute_corrective_quat(desired_quat=ref_quat, actual_quat=measured_quat)
        corrected_quat = correct_quat(measured_quat, corrective_quat)
        pos_correction = compute_translational_correction(reference_pos, measured_pos, corrected_quat)
        corrections[id] = {'pos_corr': list(pos_correction), 'quat_corr': list(corrective_quat)}

    with open(filename, mode='w') as f:
        json.dump(corrections, f)

def load_dict_from_json(fname):
    with open(fname, mode='r') as f:
        data = json.load(f)

    # convert keys to integers
    data = {int(key): data[key] for key in data}

    return data

def read_corrections(fname):
    data = load_dict_from_json(fname)
    d = dict()
    for key in data:
        d[key] = {k: np.array(data[key][k]) for k in data[key]}

    return d

def read_measured_poses(fname):
    data = load_dict_from_json(fname)
    d = dict()
    for key in data:
        d[key] = {k: np.array(data[key][k]) for k in data[key]}

    return d

def read_block_sizes(fname):
    data = load_dict_from_json(fname)
    return data



def correct_poses(measured_poses, corrections):
    corrected_poses = dict()
    for id in measured_poses:
        measured_pos = measured_poses[id]['pos']
        measured_quat = measured_poses[id]['quat']
        pos_corr = corrections[id]['pos_corr']
        quat_corr = corrections[id]['quat_corr']

        corrected_quat = correct_quat(measured_quat, quat_corr)
        corrected_pos = correct_position(measured_pos, pos_corr, corrected_quat)
        corrected_poses[id] = {'pos': corrected_pos, 'quat': corrected_quat}

    return corrected_poses


if __name__ == "__main__":
    # declarations and preparations
    cam = Camera(cam1_serial, cam1_mtx_11cm_3, cam1_dist_11cm_3)
    camera_params = cam.get_params()
    detector = dt_apriltags.Detector(nthreads=detection_threads,
                                     quad_decimate=quad_decimate,
                                     quad_sigma=quad_sigma,
                                     decode_sharpening=decode_sharpening)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1000, 1000)

    # define block sizes
    # with open('block_sizes.json', mode='r') as f:
    #     block_sizes = json.load(f)

    # block_sizes = read_block_sizes('block_poses.json')
    # d = dict()
    # for key in block_sizes:
    #     d[key] = {k: float(block_sizes[key][k]) for k in block_sizes[key]}
    #
    # with open('block_sizes.json', 'w') as f:
    #     json.dump(d, f)


    # take images and write measured poses to file
    # collect_poses(cam , camera_params, detector, 'measured_poses.json', 112)

    # compute and save corrections

    # measured_poses = read_measured_poses('measured_poses.json')
    # create_file_with_corrections(block_sizes, measured_poses, 'corrections.json')
