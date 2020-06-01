# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2
# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
    img = cv2.imread(filename)
    for result in result_list:
        x, y, width, height = result['box']
        cv2.rectangle(img, (x, y), (x+width, y+height), (255, 0, 0), -1)
    #pixels2= cv2.bitwise_not(pixels)
    cv2.imwrite("new8.jpg",img)
filename = 'cov6.jpg'
# load image from file
pixels = pyplot.imread(filename)
pixels2 = cv2.imread(filename)
#print(pixels)
#print("DIVIDE")
for i in pixels2:
    for j in i:
        temp = j[0]
        j[0] = j[2]
        j[2] = temp

#print(pixels2)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels2)
# display faces on the original image
draw_image_with_boxes(filename, faces)
