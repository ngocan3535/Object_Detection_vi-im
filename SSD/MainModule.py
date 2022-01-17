from Nhan_Dien_Thuc_Te import *

cap = cv2.VideoCapture(0)
address = "http://192.168.1.6:8080/video"
cap.open(address)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 70)

while True:
    success, img = cap.read()
    result, objectInfo = getObjects(img, 0.45, 0.2)
    # print(objectInfo)
    cv2.imshow("Output", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break