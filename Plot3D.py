import dataclasses
from typing import List, Mapping, Optional, Tuple, Union

from model_setup import Model_Setup

import matplotlib.pyplot as plt
import pandas as pd

import imageio
from pathlib import Path

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
    def __init__(self, config):
        self.config = config
        self.list_size = self.config.NUM_COORDS
        self.landmark_list = []
        for i in range(self.list_size):
            obj = Landmark()
            self.landmark_list.append(obj)

        self.Min_Max_axis = {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "z_min": None,
            "z_max": None,
        }

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

        self.Min_Max_axis["x_min"] = min(x)
        self.Min_Max_axis["x_max"] = max(x)
        self.Min_Max_axis["y_min"] = min(y)
        self.Min_Max_axis["y_max"] = max(y)
        self.Min_Max_axis["z_min"] = min(z)
        self.Min_Max_axis["z_max"] = max(z)

        self.load_xyz(x, y, z, visibility)

    def load_csv(self, RESULT_CSV):
        df = pd.read_csv(RESULT_CSV)
        self.load_df(df)
        return df


def save_image(config, csv_file, IMAGE_PATH):
    landmark_list_all = Landmark_list(config)
    df = landmark_list_all.load_csv(csv_file)
    landmark_list_all.load_df(df)
    # Plot every frame
    index = 0
    counter = 0
    for i in range(len(df)):
        if index % 24 == 0:
            landmark_list = Landmark_list(config)
            df_temp = df.iloc[i, :]
            x = df_temp[df.columns.str.startswith("x")].to_numpy().flatten()
            y = df_temp[df.columns.str.startswith("y")].to_numpy().flatten()
            z = df_temp[df.columns.str.startswith("z")].to_numpy().flatten()
            visibility = df_temp[df.columns.str.startswith("v")].to_numpy().flatten()
            landmark_list.load_xyz(x, y, z, visibility)
            plot_landmarks(
                landmark_list,
                config.POSE_CONNECTIONS,
                counter=counter,
                IMAGE_PATH=IMAGE_PATH,
                Min_Max_axis=landmark_list_all.Min_Max_axis,
            )
            counter += 1   
        index += 1


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
    counter=None,
    IMAGE_PATH=None,
    Min_Max_axis=None,
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
    if Min_Max_axis:
        ax.set_xlim3d(-1 * Min_Max_axis["z_max"], -1 * Min_Max_axis["z_min"])
        ax.set_ylim3d(Min_Max_axis["x_min"], Min_Max_axis["x_max"])
        ax.set_zlim3d(-1 * Min_Max_axis["y_max"], -1 * Min_Max_axis["y_min"])
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

        plt.savefig(IMAGE_PATH + "\\" + "fram_sec_{}.png".format(counter), dpi=50)


def save_gif(IMAGE_PATH):
    # https://medium.com/swlh/python-animated-images-6a85b9b68f86

    image_path = Path(IMAGE_PATH)
    images = list(image_path.glob("*.png"))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))

    imageio.mimwrite("animated_from_images.gif", image_list)
    """
    from pygifsicle import optimize
    gif_path = 'animated_from_video.gif'# create a new one
    optimize(gif_path, 'animated_from_video_optimized.gif')# overwrite the original one
    optimize(gif_path)
    """
