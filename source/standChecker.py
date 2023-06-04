import tensorflow as tf
import numpy as np 
import cv2
import math

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

pointLine = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

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

image_path = "./img/st1.jpg"
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)

input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, 256, 256)

model_path = "./model/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"
#model_path = "./model/movenet_lightning_fp16.tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

input_image = tf.cast(input_image, dtype=tf.uint8)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
interpreter.invoke()
keypoints = interpreter.get_tensor(output_details[0]['index'])

width = 640
height = 640

input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, width, height)
input_image = tf.cast(input_image, dtype=tf.uint8)

image_np = np.squeeze(input_image.numpy(), axis=0)
image_np = cv2.resize(image_np, (width, height))
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

for keypoint in keypoints[0][0]:
    x = int(keypoint[1] * width)
    y = int(keypoint[0] * height)

    cv2.circle(image_np, (x, y), 4, (0, 0, 255), -1)

for i in range(18):
    x1 = int(keypoints[0][0][pointLines[i][0]][1] * width)
    y1 = int(keypoints[0][0][pointLines[i][0]][0] * height)

    x2 = int(keypoints[0][0][pointLines[i][1]][1] * width)
    y2 = int(keypoints[0][0][pointLines[i][1]][0] * height)

    cv2.line(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    xDiff = abs(x2-x1)
    yDiff = abs(y2-y1)
    lineLens[i] = math.sqrt(xDiff**2 + yDiff**2)

cv2.imshow("pose estimation", image_np)
cv2.waitKey()