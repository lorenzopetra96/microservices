import numpy as np
import math
import imutils


def euclidean_distance(a, b):
    """
    Compute euclidean distance between two points
    :param a: first point coordinates
    :param b: second point coordinates
    :return: euclidean distance
    """
    return sum((np.array(a) - np.array(b)) ** 2) ** 0.5


def face_alignment_angle(left_eye, right_eye):
    """
    Compute angle for face alignment by using eyes position
    :param left_eye: left eye coordinates
    :param right_eye: right eye coordinates
    :return: angle indicating how many degrees the image should be rotated to be aligned
    """
    angle = 0

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    if left_eye_y == right_eye_y:
        return angle  # face is already aligned

    if left_eye_y > right_eye_y:
        point = (right_eye_x, left_eye_y)
        direction = 1
    else:
        point = (left_eye_x, right_eye_y)
        direction = -1

    a = euclidean_distance(left_eye, point)
    b = euclidean_distance(right_eye, point)
    c = euclidean_distance(right_eye, left_eye)

    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # rotate base image
        if direction == 1:
            angle = 90 - angle

        angle = direction * angle

    return angle


def rotate_img_and_detect_face(img, detector, threshold=0.95, n_rotations=0, rotation_degrees=90, alignment=True):
    """
    Rotate an image and detect a face.
    :param img: input image
    :param detector: detector used to detect faces
    :param threshold: minimum confidence to detect a face
    :param n_rotations: number of time the image is rotated
    :param rotation_degrees: degrees the image is rotated each time
    :param alignment: True to align the detected face by using eye coordinates
    :return: True/False if the face is detected, the bounding box and the angle rotation
    """

    best_face_info = {}
    box = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
    best_angle = 0

    n_rotations += 1  # additional rotation
    max_rotation = min(360, n_rotations * rotation_degrees)  # do not rotate more than 360 degrees

    for angle in range(0, max_rotation, rotation_degrees):
        # rotate image
        print(f'Rotating image by {angle} degrees...')
        rotated_img = imutils.rotate_bound(img, angle) if angle else img  # rotate only if angle != 0

        # detect face
        faces = detector.detect_faces(rotated_img)

        # if a good face is detected
        if faces and faces[0]['confidence'] > threshold:
            # if no previous face detected or new face is better than previous
            if not best_face_info or faces[0]['confidence'] > best_face_info['confidence']:
                best_face_info = faces[0]
                best_angle = angle
                print('New best face found')

    # if at least one face is detected
    detected = len(best_face_info)
    if detected:
        x, y, w, h = best_face_info['box']
        box = {'x': x, 'y': y, 'w': w, 'h': h}

        # use eyes keypoints to perform alignment
        if alignment:
            left_eye = best_face_info['keypoints']['left_eye']
            right_eye = best_face_info['keypoints']['right_eye']
            alignment_angle = face_alignment_angle(left_eye, right_eye)
            best_angle += alignment_angle

    return detected, box, best_angle
