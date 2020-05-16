import fastai
from fastai.vision import *

import cv2

from utils import *


def test(view, recreate_vid, recreate_labels):
    learn = load_learner("models/")

    cap = cv2.VideoCapture("data/train.mp4")
    outcap = cv2.VideoCapture(0)
    outcap.set(3, 640)
    outcap.set(4, 480)

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter("data/model-output.mp4", fourcc, 20.0, (640, 480))

    counter = 0
    while cap.isOpened():
        counter += 1

        ret, read = cap.read()
        frame = read.copy()
        frame = np.rollaxis(frame, 2, 0)
        frame = torch.from_numpy(frame)

        frame = frame.to(device=("cpu"), dtype=torch.float)

        pred = learn.predict(frame)
        speed = round((pred[2] / 10).item(), 6)

        # Render the video with the labels
        if view:
            draw_speed(read, speed, counter, True, True)
        # Recreate the mp4 file with the models predictions overlayed
        if recreate_vid:
            out.write(read)
        # cv2.imwrite(f"data/test_imgs/{speed}_{counter}.jpg", read)
        # Recreate the labels text file for the submission
        if recreate_labels:
            with open("data/model-preds.txt", "a") as fname:
                fname.write(f"{speed}\n")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test(True, True, True)
