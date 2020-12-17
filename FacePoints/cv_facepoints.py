import cv2
import urllib.request as urlreq
import os
import matplotlib.pyplot as plt
from pylab import rcParams


path = "C:/DPLMKData/FRAMES/"
user = "michael_anomal/"

def HaarCascade(image_gray):
    #haar casdae
    haarcascade = "haarcascade_frontalface_alt2.xml"
    if (haarcascade in os.listdir(os.curdir + "/FacePoints")):
        print("File exists")

    detector = cv2.CascadeClassifier("C:/Users/Pinkie/Desktop/DPLMK_nn/FacePoints/" + haarcascade)
    faces = detector.detectMultiScale(image_gray)

    # Print coordinates of detected faces
    print("Faces:\n", faces)

    for face in faces:
    #     save the coordinates in x, y, w, d variables
        (x,y,w,d) = face
        # Draw a white coloured rectangle around each face using the face's coordinates
        # on the "image_template" with the thickness of 2 
        cv2.rectangle(image_gray,(x,y),(x+w, y+d),(255, 255, 255), 2)

    plt.axis("off")
    plt.imshow(image_gray)
    plt.title('Face Detection')
    plt.waitforbuttonpress()

def DetectLandMarks(image_gray):
    LBFmodel = "lbfmodel.yaml"
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel("C:/Users/Pinkie/Desktop/DPLMK_nn/FacePoints/" + LBFmodel)

    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(image_gray, faces)

    for landmark in landmarks:
        for x,y in landmark[0]:
            # display landmarks on "image_cropped"
            # with white colour in BGR and thickness 1
            cv2.circle(image_cropped, (x, y), 1, (255, 255, 255), 1)
    plt.axis("off")
    plt.imshow(image_cropped)


def GetPoints():
    lst = len(os.listdir(path + user))
    for filenum in range(lst):
        pic = "img" + str(filenum) + ".jpg"

        image = cv2.imread(path + user + pic)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Cascade
        HaarCascade(image_gray)

GetPoints()
#HaarCascade()