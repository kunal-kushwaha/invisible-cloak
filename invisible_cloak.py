import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_red = np.array([0, 120, 150])
        u_red = np.array([10, 255, 255])

        mask = cv2.inRange(hsv, l_red, u_red)
        # cv2.imshow("mask", mask) # only red

        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
        part1 = cv2.bitwise_and(back, back, mask=mask)
        # cv2.imshow("mask", part1) # behind only the red one, else everything black

        mask = cv2.bitwise_not(mask) # all things other than the red
        # cv2.imshow("mask", mask)

        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("mask", part2) # real time feed, only red will turn black

        cv2.imshow("mask", part2 + part1)

        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
