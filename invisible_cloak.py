# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    # taking each frame
    ret, frame = cap.read()
    if ret:
        # what is hsv?
        # The HSV model describes colors similarly to how the human eye tends
        # to perceive color.
        # RGB defines color in terms of a combination of primary colors.
        # In situations where color description plays an integral role,
        # the HSV color model is often preferred over the RGB model.

        # 'Hue' represents the color
        # 'Saturation' represents the amount to which that respective color is
        #mixed with white
        # 'Value' represents the amount to which that respective color is
        # mixed with black (Gray level).

        # In RGB, we cannot separate color information from luminance.
        # HSV or Hue Saturation Value is used to separate image
        # luminance from color information.
        # (luminance is intensity of light)

        # convert bgr to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv", hsv)

        # how to get the hsv value?
        # [H-10, 100,100] and [H+10, 255, 255] as lower bound and upper bound
        # red = np.uint8([[[0,0,255]]])
        # hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
        # print(hsv_red)

        # define range of red color in hsv
        l_red = np.array([0, 120, 120])
        u_red = np.array([10, 255, 255])

        # threshold the hsv value to get only blue colors
        mask = cv2.inRange(hsv, l_red, u_red)
        # cv2.imshow("mask", mask) # only red

        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
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
