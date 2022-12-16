from apiflask import Schema
from apiflask.fields import String, Integer, Float, Boolean, Nested


class FaceDetectionInput(Schema):
    pic = String(required=True)
    threshold = Float(required=False)
    n_rotations = Integer(required=False)
    rotation_degrees = Integer(required=False)
    alignment = Boolean(required=False)


class BoundingBox(Schema):
    x = Integer()
    y = Integer()
    w = Integer()
    h = Integer()


class FaceDetectionOutput(Schema):
    detected = Boolean()
    box = Nested(BoundingBox)
    angle = Integer()

