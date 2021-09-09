import dataclasses
from typing import List, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

NUM_COORDS = 33

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3


class Landmark:
    def __init__(self, x=None, y=None, z=None, visibility=None):

        self.x = None
        self.y = None
        self.z = None
        self.visibility = None


class Landmark_list:
    def __init__(self, list_size=None):

        self.list_size = list_size
        self.landmark_list = []
        for i in range(list_size):
            obj = Landmark()
            self.landmark_list.append(obj)

    def load_xyz(self, x, y, z, visibility):
        for i, landmark in enumerate(self.landmark_list):
            self.landmark_list[i].x = x[i]
            self.landmark_list[i].y = y[i]
            self.landmark_list[i].z = z[i]
            self.landmark_list[i].visibility = visibility[i]

    def load_df(self, df):

        x = df.loc[:, df.columns.str.startswith("x")].to_numpy().flatten()
        y = df.loc[:, df.columns.str.startswith("y")].to_numpy().flatten()
        z = df.loc[:, df.columns.str.startswith("z")].to_numpy().flatten()
        visibility = df.loc[:, df.columns.str.startswith("v")].to_numpy().flatten()

        self.load_xyz(x, y, z, visibility)

    def load_csv(self, RESULT_CSV):
        df = pd.read_csv(RESULT_CSV)
        self.load_df(df)
        return df

# Adopt the plot_landmarks from MediaPipe
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalize_color(color):
    return tuple(v / 255.0 for v in color)


def plot_landmarks(
    landmark_list,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR, thickness=5),
    connection_drawing_spec: DrawingSpec = DrawingSpec(color=BLACK_COLOR, thickness=5),
    elevation: int = 10,
    azimuth: int = 10,
):
    """Plot the landmarks and the connections in matplotlib 3d.
    Args:
      landmark_list: A normalized landmark list proto message to be plotted.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected.
      landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color and line thickness.
      connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.
      elevation: The elevation from which to view the plot.
      azimuth: the azimuth angle to rotate the plot.
    Raises:
      ValueError: If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark_list):
        if landmark.visibility and landmark.visibility < _VISIBILITY_THRESHOLD:
            continue
        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness,
        )
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        num_landmarks = landmark_list.list_size
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness,
                )

        plt.show()
