import cv2
from checkPose import *

if __name__=='__main__':
    # openpose model - body25
    protoFile = "model/body25/pose_deploy.prototxt"
    modelFile = "model/body25/pose_iter_584000.caffemodel"

    imgPath = "img/pose6.jpg"

    # 키포인트를 저장할 빈 리스트
    joints = []

    # 이미지 읽어오기
    img = cv2.imread(imgPath)
    img, joints = jointsOutput(img=img, bodyParts=bodyParts, proto=protoFile, model=modelFile, threshold=0.2)
    linesOutput(img, bodyPairs, joints)
    