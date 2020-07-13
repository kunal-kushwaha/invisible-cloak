import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, back = cap.read()
    if ret==True:
        cv2.imshow("image6.jpg", back)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.imwrite('image.jpg', back)
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

