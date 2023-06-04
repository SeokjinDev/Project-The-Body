import cv2

def centerLoc(points, triPoints):
    x = sum([points[i][0] for i in triPoints]) // 3
    y = sum([points[i][1] for i in triPoints]) // 3
    return x, y

def faceDir(img, facePoints, correction=0):
    outCenter = centerLoc(facePoints, [0, 17, 18]) # 0: "Nose", 17: "REar", 18: "LEar",
    inCenter = centerLoc(facePoints, [0, 15, 16])  # 0: "Nose", 15: "REye", 16: "LEye"

    cv2.circle(img, (outCenter[0], outCenter[1]), 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(img, str("out"), (outCenter[0], outCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.circle(img, (inCenter[0], inCenter[1]), 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(img, str("in"), (inCenter[0], inCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

    #print(outCenter)
    #print(inCenter)

    if outCenter[0] - inCenter[0] > correction:
        cv2.putText(img, "right", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    elif outCenter[0] - inCenter[0] < correction:
        cv2.putText(img, "left", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    else:
        cv2.putText(img, "center", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    
    if outCenter[1] - inCenter[1] > correction:
        cv2.putText(img, "up", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    elif outCenter[1] - inCenter[1] < correction:
        cv2.putText(img, "down", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    else:
        cv2.putText(img, "center", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    cv2.imshow("Face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return outCenter, inCenter