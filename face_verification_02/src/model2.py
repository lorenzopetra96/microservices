import sys
from waitress import serve
import numpy as np
import face_recognition_models
import dlib
from datetime import datetime
from pathlib import Path
from apiflask import APIFlask, HTTPError
from utils import kde_utils, conversion_utils
from model.FaceVerificationModels import FaceVerificationInput, FaceVerificationOutput

# kde of true and false matches
root_dir = Path(globals().get("__file__", "./_")).absolute().parents[1].__str__()
dataframe_src = root_dir + '/df/face_recognition_v3.csv'
kde_true, kde_false = kde_utils.get_kde(dataframe_src)

# flask
app = APIFlask(__name__)

# pose predictor model
predictor_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor = dlib.shape_predictor(predictor_model)
# face recognition model
face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def face_encodings(face_image, face_locations):
    """
    Compute face encoding
    :param face_image: image containing a face
    :param face_locations:
    :return:
    """
    raw_landmarks = [pose_predictor(face_image, face_location) for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, 1))
            for raw_landmark_set in raw_landmarks]


def face_distance(face_encoding1, face_encoding2):
    """
    Compute distance between two faces
    :param face_encoding1: first face encoding
    :param face_encoding2: second face encoding
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    return np.linalg.norm(face_encoding1 - face_encoding2)


def compute_distance(img1, img2):
    """
    Compute distance between faces in two images
    :param img1: first image
    :param img2: second image
    :return: distance between faces
    """
    print('Using face_recognition...')

    # tuple in (left, top, right, bottom)
    face_locations1 = [dlib.rectangle(0, 0, img1.shape[1], img1.shape[0])]
    face_locations2 = [dlib.rectangle(0, 0, img2.shape[1], img2.shape[0])]

    # face encoding
    # pass face locations to avoid (re)computing face detection
    encoding1 = face_encodings(img1, face_locations=face_locations1)[0]
    encoding2 = face_encodings(img2, face_locations=face_locations2)[0]
    distance = face_distance(encoding1, encoding2)
    return distance


@app.route('/faceVerification', methods=["POST"])
@app.doc(summary="Perform face verification",
         description="Returns the confidence two faces are the same person",
         responses={200: "Verification performed", 400: "Image(s) not valid"})
@app.input(FaceVerificationInput, location='json')
@app.output(FaceVerificationOutput)
def face_verification(json):
    time_start = datetime.now()
    print("Start time: ", time_start)

    doc_pic = json['docPic']
    usr_pic = json['usrPic']

    try:
        doc_pic = conversion_utils.b64_to_numpy(doc_pic)  # , "model2_img1.jpg")
        usr_pic = conversion_utils.b64_to_numpy(usr_pic)  # , "model2_img2.jpg")
    except:
        return HTTPError(400, message='Image(s) not valid')

    distance = compute_distance(doc_pic, usr_pic)
    print(f'distance: {distance: .2f}')

    confidence = kde_utils.compute_confidence(distance, kde_true, kde_false)
    print(f'confidence: {confidence: .5f}')

    time_end = datetime.now()
    print("End time: ", time_end)
    print("Elapsed time: ", time_end - time_start)

    return {"confidence": confidence}


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Running (default)...')
        # app.run(host='127.0.0.1', port=5002) # development
        serve(app, host="0.0.0.0", port=5002)  # production
    else:
        print('Running...')
