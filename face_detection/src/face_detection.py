import sys
from apiflask import APIFlask, HTTPError
from mtcnn_cv2 import MTCNN
from utils import detection_utils, conversion_utils
import json
from model.FaceDetectionModels import FaceDetectionInput, FaceDetectionOutput
from waitress import serve

# flask
app = APIFlask(__name__)


def check_params(params, threshold=0.90, n_rotations=0, rotation_degrees=90, alignment=True):
    """
    Check if input parameters are valid. If not, convert the invalid ones to their default value
    :param params: input parameters
    :param threshold: confidence threshold for face detection
    :param n_rotations: number of rotations to perform
    :param rotation_degrees: amount of degrees for each rotation
    :param alignment: True to perform face alignment
    :return: valid parameters
    """
    try:
        # threshold in [0; 1]
        threshold = max(0.0, min(1.0, float(params['threshold']))) if 'threshold' in params else threshold
    except:
        threshold = threshold

    try:
        # n_rotations >= 0
        n_rotations = max(0, int(params['n_rotations'])) if 'n_rotations' in params else n_rotations
    except:
        n_rotations = n_rotations

    try:
        # rotation degrees in [1; 359]
        rotation_degrees = max(1, min(359, int(params['rotation_degrees']))) if 'rotation_degrees' in params \
            else rotation_degrees
    except:
        rotation_degrees = rotation_degrees

    try:
        alignment = json.loads(params['alignment']) if 'alignment' in params else alignment
    except:
        alignment = alignment

    return threshold, n_rotations, rotation_degrees, alignment


@app.route('/faceDetection', methods=["POST"])
@app.doc(summary="Perform face detection",
         description="Returns if a face is detected, and eventually the bounding box and the rotation angle",
         responses={200: "Detection performed", 400: "Image not valid"})
@app.input(FaceDetectionInput, location='json')
@app.output(FaceDetectionOutput)
def face_detection(json):
    pic = json['pic']
    try:
        pic = conversion_utils.b64_to_numpy(pic)
    except:
        HTTPError(400, message='Image not valid')

    threshold, n_rotations, rotation_degrees, alignment = check_params(json)

    print(f"Using params:\nthreshold: {threshold}\nn_rotations: {n_rotations}\nrotation_degrees: {rotation_degrees}\n"
          f"alignment: {alignment}")

    detected, box, angle = detection_utils.rotate_img_and_detect_face(pic, detector, threshold, n_rotations,
                                                                      rotation_degrees, alignment)

    response = {
        'detected': "true" if detected else "false",  # True/False -> "true"/"false"
        'box': box,
        'angle': angle
    }

    return response


if __name__ == '__main__':
    detector = MTCNN(min_face_size=50)

    if len(sys.argv) == 1:
        print('Running (default)...')
        # app.run(host='127.0.0.1', port=5004)  # development
        serve(app, host="0.0.0.0", port=5004)  # production
    else:
        print('Running...')

