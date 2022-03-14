import tensorflow as tf
import numpy as np
import os
import json

import pandas as pd

from rich.progress import Progress

from utils import GDriveDownloader
from utils import SpatialProjection
from utils import DataLoader

from model import ProjectionNet

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

EXPERIMENT = "DYNAMIC"  # STATIC or DYNAMIC


# ... config
config_file = open("config/config.json")
config = json.load(config_file)
config_file.close()

gestures = None
if EXPERIMENT == "STATIC":
    gestures = config["static_gestures"]
elif EXPERIMENT == "DYNAMIC":
    gestures = config["dynamic_gestures"]
else:
    raise ValueError("Wrong experiment!")

projection = SpatialProjection(
    n_points=config["segment_len"],
    width=config["line_width"],
    dt=config["dt"]
)

# ... Download files
downloader = GDriveDownloader()
# downloader.download(
#     fid=config["dataset_id"],
#     destination=config["data_dir"]
# )

dataloader = DataLoader(
    data_dir=config["data_dir"],
    images_dir=config["images_dir"],
    channels_dir=config["channels_dir"],
    users=config["users"],
    gestures=gestures
)

dataloader.extract_channels(
    fs=config["fs"],
    imu_cutoff=config["imu_cutoff"],
    window_len=config["segment_len"]
)

dataloader.generate_projection_images(projection)

train_ds, test_ds = dataloader.load_ds(
    test_subjects=["001"],
    image_shape=(config["img_size"], config["img_size"]),
    batch_size=config["batch_size"]
)

print(train_ds.element_spec)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(config["img_size"], config["img_size"], 3),
    include_top=False,
    weights='imagenet'
)

model = ProjectionNet(
    img_size=config["img_size"],
    segment_len=config["segment_len"],
    n_classes=len(gestures),
    base_model=base_model,
    n_projections=3,
    n_channels=5
)
model(next(iter(train_ds)))
model.summary()
