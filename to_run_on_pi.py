
import cv2
import pathlib
import simple_pid


# draws rectangles around all faces
# but returns the rectangle around the largest face
def draw_rectangles_around_face(frame):
    casecade_path = pathlib.Path(__file__).parent.absolute().joinpath(
        'haarcascade_frontalface_default.xml')
    clf = cv2.CascadeClassifier(str(casecade_path))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2, minSize=(
        30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # print('Faces found: ', len(faces))
    if (len(faces) != 1):
        return None
    # draw rectangle around face
    x, y, width, height = faces[0]
    cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
    return (x, y, width, height)


def main():
    print("Starting camera...")
    cap = cv2.VideoCapture(1)
    # get resolution of camera
    center_frame_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    center_frame_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    pid1 = simple_pid.PID(0.1, 0.01, 0.05, setpoint=center_frame_x)
    pid2 = simple_pid.PID(0.1, 0.01, 0.05, setpoint=center_frame_y)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        _, frame = cap.read()
        tup = draw_rectangles_around_face(frame)
        if tup == None:
            continue

        # draw line to center of frame

        center_rectangle_x = tup[0] + int(tup[2] / 2)
        center_rectangle_y = tup[1] + int(tup[3] / 2)
        cv2.line(frame, (center_frame_x, center_frame_y), (center_rectangle_x, center_rectangle_y), (0, 0, 255), 2)
        cv2.imshow('frame', frame)

        print("x: ", center_rectangle_x, "y: ", center_rectangle_y)
        pid1_out = pid1(center_rectangle_x)
        pid2_out = pid2(center_rectangle_y)
        print("pid1: ", pid1_out, "pid2: ", pid2_out)



    cap.release()


main()
