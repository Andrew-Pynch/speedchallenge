import sys

import cv2
import numpy as numpy

cap = cv2.VideoCapture("data/train.mp4")
file = "data/train.txt"


def getSpeedList(file):
    with open(file) as f:
        lines = [line.rstrip("\n") for line in f]

    return lines


def drawSpeed(frame, speed, counter, show_counter, show_percentage):
    font = cv2.FONT_HERSHEY_SIMPLEX

    if show_counter:
        cv2.putText(
            frame,
            f"{round(float(speed), 2)}MPH",
            (475, 25),
            font,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    if show_percentage:
        cv2.putText(
            frame,
            f"Video %{(counter/20400)*100}",
            (450, 75),
            font,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    print(speed)
    cv2.imshow("frame", frame)


def playVid(vid, label_file):
    labels = getSpeedList(label_file)
    counter = 0
    while cap.isOpened():
        counter += 1
        speed = labels[counter]

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        drawSpeed(gray, speed, counter, True, True)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    playVid(cap, labels)


def get(item):
    try:
        print(len(item))
    except:
        print("cant get len of item")
    try:
        print(item.shape)
    except:
        print("cant get shape of item")
    try:
        print(type(item))
    except:
        print("cant get type of item")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Run\n'python utils.py view'\nto view the dataset with some cool metrics")
    elif sys.argv[1] == "view":
        """This plays the video with the current speed and frame counter"""
        playVid(cap, file)
