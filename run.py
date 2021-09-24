import os
from model_setup import Model_Setup
from PoseDataGenerator import PoseDataGenerator
from Plot3D import save_image,save_gif,save_video
from Train import Train
import pandas as pd

# Folders
# Video folders to use for trainig
PATH_VIDEO = 'videos'
# CSV pose results of video folders
RESULT_CSV = 'results'
# produce the image for each N frames of csv file
IMAGE_PATH ='images'
# produce the image for each N frames of generate.csv
GENERATE_IMAGE_PATH ='generate_images'
# initial CSV pose as an input for generation
INIT_CSV = 'initial'
# The video that use for generation
PATH_VIDEO_GEN = 'generator'
# File of moddel that train has been done
MODEL_PATH = 'autoregression'

# Run parts
TRAIN = False
CREATE_IMAGES = False
CREATE_GIF = False
CREATE_VIDEO = False
POSE_GENERATION = False
CREATE_IMAGES_GEN = True
CREATE_VIDEO_GEN = True
GENERATE_NEW_DANCE_CSV = True
# Model config
config = Model_Setup()
config.HIST_WINDOW = 10*24
config.MODEL_NAME = MODEL_PATH
config.HDF = "data.h5"
config.CREATE_HDF = False
# Generate training data from Video Clip
pose_data = PoseDataGenerator(config)
if POSE_GENERATION:
    if os.path.isfile(RESULT_CSV) == False:
        pose_data.generate_pose_multifile(PATH_VIDEO,RESULT_CSV)

# Testing 3DPlot Pose Estimation
if CREATE_IMAGES:
    save_image(config,RESULT_CSV+'\coordinates_1.csv',IMAGE_PATH)

if CREATE_GIF: 
    save_gif(IMAGE_PATH)

if CREATE_VIDEO: 
    save_video(IMAGE_PATH)

# Train
train_obj = Train(config)
if TRAIN:
    train_obj.fit(RESULT_CSV)
    train_obj.plot_performance()

#Generator

if GENERATE_NEW_DANCE_CSV:
    #pose_data.generate_pose_multifile(PATH_VIDEO_GEN,INIT_CSV)
    df_init = pd.read_csv(INIT_CSV+'/'+'coordinates.csv')

    df_init = train_obj.dataset_df(df_init)
    train_obj.generator(MODEL_PATH,df_init,frames_future=1000)

# Animation Generator
if CREATE_IMAGES_GEN:
    save_image(config,'generate.csv',GENERATE_IMAGE_PATH)
    #save_gif(GENERATE_IMAGE_PATH)

if CREATE_VIDEO_GEN:
    save_video(GENERATE_IMAGE_PATH)