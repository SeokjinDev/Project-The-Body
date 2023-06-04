import tensorflow as tf
import numpy as np 
import cv2
import math


# Init
partPointsLoc = {
    0: 0, # 코
    1: 0, # 왼쪽 눈
    2: 0, # 오른쪽 눈
    3: 0, # 왼쪽 귀
    4: 0, # 오른쪽 귀
    5: 0, # 왼쪽 어깨
    6: 0, # 오른쪽 어깨
    7: 0, # 왼쪽 팔꿈치
    8: 0, # 오른쪽 팔꿈치
    9: 0, # 왼쪽 손목
    10: 0,# 오른쪽 손목
    11: 0,# 왼쪽 골반
    12: 0,# 오른쪽 골반
    13: 0,# 왼쪽 무릎
    14: 0,# 오른쪽 무릎
    15: 0,# 왼쪽 발목
    16: 0 # 오른쪽 발목
    }

pointLines = {
    0 : [0, 1], # 코    -왼쪽눈
    1 : [0, 2], # 코    -오른쪽눈
    2 : [1, 3], # 왼쪽눈-왼쪽귀
    3 : [2, 4], # 오른쪽눈-오른쪽귀
    4 : [0, 5], # 코    -왼쪽어깨
    5 : [0, 6], # 코    -오른쪽어깨
    6 : [5, 7], # 왼쪽어깨-왼쪽팔꿈치
    7 : [7, 9], # 왼쪽팔꿈치-왼쪽손목
    8 : [6, 8], # 오른쪽어깨-오른쪽팔꿈치
    9 : [8, 10],# 오른쪽팔꿈치-오른쪽손목
    10 :[5, 6], # 왼쪽어깨-오른쪽어깨
    11 :[5, 11],# 왼쪽어깨-왼쪽골반
    12 :[6, 12],# 오른쪽어깨-오른쪽골반
    13 :[11, 12],#왼쪽골반-오른쪽골반
    14 :[11, 13],#왼쪽골반-왼쪽무릎
    15 :[13, 15],#왼쪽무릎-왼쪽발목
    16 :[12, 14],#오른쪽골반-오른쪽무릎
    17 :[14, 16] #오른쪽무릎-오른쪽발목
}

lineLens = {
    0 : 0, # 코    -왼쪽눈
    1 : 0, # 코    -오른쪽눈
    2 : 0, # 왼쪽눈-왼쪽귀
    3 : 0, # 오른쪽눈-오른쪽귀
    4 : 0, # 코    -왼쪽어깨
    5 : 0, # 코    -오른쪽어깨
    6 : 0, # 왼쪽어깨-왼쪽팔꿈치
    7 : 0, # 왼쪽팔꿈치-왼쪽손목
    8 : 0, # 오른쪽어깨-오른쪽팔꿈치
    9 : 0,# 오른쪽팔꿈치-오른쪽손목
    10 :0, # 왼쪽어깨-오른쪽어깨
    11 :0,# 왼쪽어깨-왼쪽골반
    12 :0,# 오른쪽어깨-오른쪽골반
    13 :0,#왼쪽골반-오른쪽골반
    14 :0,#왼쪽골반-왼쪽무릎
    15 :0,#왼쪽무릎-왼쪽발목
    16 :0,#오른쪽골반-오른쪽무릎
    17 :0 #오른쪽무릎-오른쪽발목
}

location = {}


# Function
def valueCheck(img_path, model_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)

    input_img = tf.expand_dims(img, axis=0)
    input_img = tf.image.resize_with_pad(input_img, 256, 256)

    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_img = tf.cast(input_img, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_img.numpy())
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])

    width = 640
    height = 640

    input_img = tf.expand_dims(img, axis=0)
    input_img = tf.image.resize_with_pad(input_img, width, height)
    input_img = tf.cast(input_img, dtype=tf.uint8)

    img_np = np.squeeze(input_img.numpy(), axis=0)
    img_np = cv2.resize(img_np, (width, height))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    loc = {}

    for keypoint, i in zip(keypoints[0][0], range(17)):
        x = int(keypoint[1] * width)
        y = int(keypoint[0] * height)
        loc[i] = [x, y]

        cv2.circle(img_np, (x, y), 4, (0, 0, 255), -1)

    locDis = {}

    for i in range(18):
        x1 = int(keypoints[0][0][pointLines[i][0]][1] * width)
        y1 = int(keypoints[0][0][pointLines[i][0]][0] * height)

        x2 = int(keypoints[0][0][pointLines[i][1]][1] * width)
        y2 = int(keypoints[0][0][pointLines[i][1]][0] * height)

        cv2.line(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        xDiff = abs(x2-x1)
        yDiff = abs(y2-y1)
        locDis[i] = math.sqrt(xDiff**2 + yDiff**2)

    cv2.imshow("pose estimation", img_np)
    cv2.waitKey()
    return loc, locDis

def centerValue(a, b, c):
    return [(a[0]+b[0]+c[0])/3, (a[1]+b[1]+c[1])/3]

def ratioCheck(loc, distances):
    headCorrection = (lambda a, b: [a[0]-b[0], a[1]-b[1]])(centerValue(loc[0], loc[1], loc[2]), centerValue(loc[0], loc[3], loc[4]))
    rightUpperArmRatio = distances[8] / distances[10]
    rightForearmRatio = distances[9] / distances[8]
    return [headCorrection, rightUpperArmRatio, rightForearmRatio]

if __name__ == '__main__':
    location, locationDistance = valueCheck(img_path = "./img/st1.jpg",
                                            model_path = "./model/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite")
    ratioCheck(location, locationDistance)
    print(ratioCheck(location, locationDistance))