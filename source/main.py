import cv2
from checkPose import *
from checkFaceDir import faceDir

if __name__=='__main__':
    # Openpose model - body25
    protoFile = "model/body25/pose_deploy.prototxt"
    modelFile = "model/body25/pose_iter_584000.caffemodel"

    imgPath = "img/pose6.jpg"

    # Joints List
    joints = []

    # Read Image
    img = cv2.imread(imgPath)
    img, joints = jointsOutput(img=img, bodyParts=bodyParts, proto=protoFile, model=modelFile, threshold=0.2)
    #linesOutput(img, bodyPairs, joints)
    faceDir(img=img, facePoints=joints)