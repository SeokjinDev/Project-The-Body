import cv2

bodyParts = {0: "Nose", 1: "Neck",
            2: "RShoulder", 3: "RElbow", 4: "RWrist",
            5: "LShoulder", 6: "LElbow", 7: "LWrist",
            8: "MidHip",
            9: "RHip", 10: "RKnee", 11: "RAnkle",
            12: "LHip", 13: "LKnee", 14: "LAnkle",
            15: "REye", 16: "LEye", 17: "REar", 18: "LEar",
            19: "LBigToe", 20: "LSmallToe", 21: "LHeel",
            22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
            25: "Background"}

bodyPairs = [[0, 1], [0, 15], [0, 16],
            [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13],
            [2, 3], [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
            [11, 24], [22, 24], [23, 24]]

def jointsOutput(img, bodyParts, proto, model, threshold):
    network = cv2.dnn.readNetFromCaffe(proto, model)
    imageHeight, imageWidth = 368, 368

    # Preprocessing
    inputBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

    # Input to network
    network.setInput(inputBlob)

    # Result
    output = network.forward()
    outHeight = output.shape[2]
    outWidth = output.shape[3]

    # Original Image Size
    imgHeight, imgWidth = img.shape[:2]

    # Init
    joints = []
    for i in range(len(bodyParts)):

        # Confidence Map
        confidenceMap = output[0, i, :, :]

        # Minimum Value, Maximum Value, Minimum Location, Maximum Location
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(confidenceMap)

        x = int((imgWidth * maxLoc[0]) / outWidth)
        y = int((imgHeight * maxLoc[1]) / outHeight)
        if maxVal > threshold:  # Positioned
            cv2.circle(img, (x, y), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            joints.append((x, y))

        else:  # Not Positioned
            cv2.circle(img, (x, y), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)
            joints.append(None)

    # Image Output
    # cv2.imshow("Joints", img)
    # cv2.waitKey(0)
    return img, joints

def linesOutput(img, bodyPairs, joints):
    for pair in bodyPairs:
        part_a = pair[0]  # Head: 0
        part_b = pair[1]  # Neck: 1
        if joints[part_a] and joints[part_b]:
            cv2.line(img, joints[part_a], joints[part_b], (0, 255, 0), 3, cv2.LINE_8)
    #cv2.imshow("Lines", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()