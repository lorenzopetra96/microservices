import sys
from waitress import serve
from deepface import DeepFace
from datetime import datetime
from pathlib import Path
from apiflask import APIFlask, HTTPError
from utils import kde_utils, conversion_utils
from model.FaceVerificationModels import FaceVerificationInput, FaceVerificationOutput


# build model
model_name = 'Facenet512'
model = DeepFace.build_model(model_name)

# get kde of true and false matches
root_dir = Path(globals().get("__file__", "./_")).absolute().parents[1].__str__()
dataframe_src = root_dir + '/df/Facenet512_v3.csv'
kde_true, kde_false = kde_utils.get_kde(dataframe_src)

app = APIFlask(__name__)


def compute_distance(img1, img2, detector='skip', metric='cosine'):
    """
    Compute distance between two images by using FaceNet model
    :param img1: first image
    :param img2: second image
    :param detector: detector
    :param metric: distance metric
    :return: distance between two images
    """
    print(f'using {model_name}...')

    result = DeepFace.verify(img1, img2, model=model, detector_backend=detector, distance_metric=metric)
    distance = result['distance']
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
        doc_pic = conversion_utils.b64_to_numpy(doc_pic)  # , "model3_img1.jpg")
        usr_pic = conversion_utils.b64_to_numpy(usr_pic)  # , "model3_img2.jpg")
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
        # app.run(host='127.0.0.1', port=5003) # development
        serve(app, host="0.0.0.0", port=5003)  # production
    else:
        print('Running...')
