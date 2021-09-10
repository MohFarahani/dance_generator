import csv
import os

import cv2
import mediapipe as mp
import numpy as np

from model_setup import Model_Setup


class PoseDataGenerator:
    def __init__(self, CONFIG):
        # visualing poses
        self.mp_drawing = mp.solutions.drawing_utils
        # importing pose estimation model
        self.mp_pose = mp.solutions.pose
        # holistic
        self.mp_holistic = mp.solutions.holistic
        # CONFIG
        self.config = CONFIG

    def pose_detection(self, PATH_VIDEO, RESULT_CSV):
        index = (
            PATH_VIDEO.split("/")[-1]
            if PATH_VIDEO.split("/")[-1] != ""
            else PATH_VIDEO.split("/")[-2]
        )
        cap = cv2.VideoCapture(PATH_VIDEO)
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:

            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make Detections
                results = holistic.process(image)

                # pose_landmarks, left_hand_landmarks, right_hand_landmarks

                if self.config.SHOW_WINDOW:
                    # Recolor image back to BGR for rendering
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # 1. Pose Detections
                    self.mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        self.mp_holistic.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(
                            color=(245, 117, 66), thickness=2, circle_radius=4
                        ),
                        self.mp_drawing.DrawingSpec(
                            color=(245, 66, 230), thickness=2, circle_radius=2
                        ),
                    )

                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(
                        np.array(
                            [
                                [
                                    landmark.x,
                                    landmark.y,
                                    landmark.z,
                                    landmark.visibility,
                                ]
                                for landmark in pose
                            ]
                        ).flatten()
                    )

                    # Append class name
                    pose_row.insert(0, index)
                    # Export to CSV
                    with open(RESULT_CSV, mode="a", newline="") as f:
                        csv_writer = csv.writer(
                            f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                        )
                        csv_writer.writerow(pose_row)

                except:
                    pass
                if self.config.SHOW_WINDOW:
                    cv2.imshow("press 'q' to quit", image)
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break
            cap.release()

    def create_csv(self, RESULT_CSV):
        if os.path.isfile(RESULT_CSV) == False:
            landmarks = ["clip"]
            for val in range(self.config.NUM_COORDS):
                landmarks += [
                    "x{}".format(val),
                    "y{}".format(val),
                    "z{}".format(val),
                    "v{}".format(val),
                ]
            with open(RESULT_CSV, mode="w", newline="") as f:
                csv_writer = csv.writer(
                    f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                csv_writer.writerow(landmarks)

    def generate_pose(self, PATH_VIDEO, RESULT_EXCELL):
        self.create_csv(RESULT_EXCELL)
        self.pose_detection(PATH_VIDEO, RESULT_EXCELL)
