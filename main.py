import math

import cv2

video = cv2.VideoCapture('./videos/input.mp4')
i = 0
frameRate = video.get(5)
while video.isOpened():
    frameId = video.get(1)
    ret, frame = video.read()
    if ret == False:
        break
    if frameId % math.floor(frameRate) == 0:
        cv2.imwrite('images/img'+str(i)+'.jpg', frame)
    i+=1
video.release()
cv2.destroyAllWindows()
