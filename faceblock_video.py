from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

detector = MTCNN()
capture = cv2.VideoCapture('video2.mp4')
w1 = capture.get(3)
h1 = capture.get(4)
out = cv2.VideoWriter('output3.mp4', -1, 20.0, (int(w1),int(h1)))

while (capture.isOpened()):
    ret, image = capture.read()
    if(ret==False):
        out.release()
        break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces2 = detector.detect_faces(image)
    """
    for (x,y,w,h) in faces:
        cv2.circle(image,(x+int(w/2),y+int(h/2)),(int(h)),(255,0,0),4)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
    """
    for face in faces2:
        x,y,width,height = face['box']
        cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), -1)
    cv2.imshow('Image', image)
    out.write(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the window
capture.release()
out.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()
