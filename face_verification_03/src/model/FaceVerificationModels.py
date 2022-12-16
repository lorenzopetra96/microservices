from apiflask import Schema
from apiflask.fields import String, Float


class FaceVerificationInput(Schema):
    docPic = String(required=True)
    usrPic = String(required=True)


class FaceVerificationOutput(Schema):
    confidence = Float()
