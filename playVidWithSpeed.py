import cv2
import numpy as np

cap = cv2.VideoCapture('data/train.mp4')
file = 'data/train.txt'


def getSpeedList(file):
    with open(file) as f:
        lines = [line.rstrip('\n') for line in f]

    return lines


def drawSpeed(frame, speed, counter, showCounter, showPercentage):
    font = cv2.FONT_HERSHEY_SIMPLEX

    if (showCounter):
        cv2.putText(frame, f'{round(float(speed), 2)}MPH', (475, 25), font, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)
    if (showPercentage):
        cv2.putText(frame, f'Video %{(counter/20400)*100}', (450, 75), font, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)

    print(speed)
    cv2.imshow('frame', frame)


def playVid(vid, labels):
    counter = 0
    while(cap.isOpened()):
        counter += 1
        speed = labels[counter]

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        drawSpeed(gray, speed, counter, True, True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


labels = getSpeedList(file)
playVid(cap, labels)
