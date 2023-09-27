# use open cv to display what's on localhost:3000

import pathlib
import time
import cv2
import socket
import pickle
import numpy

def show():
    camera = cv2.VideoCapture('http://localhost:3000')
    while True:
        ret, frame = camera.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('frame', frame)

def poo():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ip = 'localhost'
    port = 3005
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        x_as_bytes = pickle.dumps(buffer)
        s.sendto((x_as_bytes), (ip, port))


def main():
    # camera = cv2.VideoCapture('http://localhost:3000')
    camera = cv2.VideoCapture(1)
    casecade_path = pathlib.Path(__file__).parent.absolute().joinpath(
    'haarcascade_frontalface_default.xml')
    clf = cv2.CascadeClassifier(str(casecade_path))
    while True:
        ret, frame = camera.read()
        # sleep 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = clf.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        print('Faces found: ', len(faces))

        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

        # save frame as png
        cv2.imwrite('frame.png', frame)

        cv2.imshow('frame', frame)


# show()
# main()
poo()
