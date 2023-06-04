import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import math

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

def readData(path):
    # temprorary code
    f = open(path, 'r', encoding='UTF8')
    data = []
    for i in f.readlines():
        data.append(float(i))
    f.close()
    return data

def faceDir(img, inner, outer, correction):
    i = [sum([a[0] for a in inner])/3, sum([a[1] for a in inner])/3]
    o = [sum([a[0] for a in outer])/3, sum([a[1] for a in outer])/3+correction[1]]

    middleValue = 10

    if i[0] - (o[0] + correction[0]) + middleValue < 0:
        horText = "right"
    elif i[0] - (o[0] + correction[0]) - middleValue > 10:
        horText = "left"
    else:
        horText = "middle"

    if i[1] - (o[1] + correction[1]) + middleValue < 0:
        verText = "up"
    elif i[1] - (o[1] + correction[1]) - middleValue > 10:
        verText = "down"
    else:
        verText = "middle"
    
    img = cv2.putText(img, horText, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
    img = cv2.putText(img, verText, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def tracking(data):
    model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
    movenet = model.signatures['serving_default']

    threshold = .3

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Error loading video')
        quit()

    success, img = cap.read()

    if not success:
        print('Error reding frame')
        quit()

    y, x, _ = img.shape

    while success:
        tf_img = cv2.resize(img, (256,256))
        tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
        tf_img = np.asarray(tf_img)
        tf_img = np.expand_dims(tf_img,axis=0)

        image = tf.cast(tf_img, dtype=tf.int32)

        outputs = movenet(image)
        keypoints = outputs['output_0']

        loc = {}
        for k, i in zip(keypoints[0,0,:,:], range(17)):
            k = k.numpy()

            if k[2] > threshold:
                yc = int(k[0] * y)
                xc = int(k[1] * x)

                loc[i] = [xc, yc]
                
                img = cv2.circle(img, (xc, yc), 2, (0, 255, 0), 5)
                img = cv2.putText(img, str(i), (xc, yc), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)

        locDis = {}

        for i in range(18):
            s, e = pointLines[i]
            if loc.get(s) is not None and loc.get(e) is not None:
                x1, y1 = loc[s]
                x2, y2 = loc[e]
                
                xDiff = abs(x2-x1)
                yDiff = abs(y2-y1)
                locDis[i] = math.sqrt(xDiff**2 + yDiff**2)

                img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        try:
            img = faceDir(img, [loc[0], loc[1], loc[2]], [loc[0], loc[3], loc[4]], [data[0], data[1]])
        except:
            pass
        
        cv2.imshow('Movenet', img)
        if cv2.waitKey(1) == ord("q"):
            break

        success, img = cap.read()
    cap.release()

if __name__ == '__main__':
    data = readData('./data/ratio.txt')
    tracking(data)