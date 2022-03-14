import tensorflow as tf
import numpy as np
import os
import json

import pandas as pd

from rich.progress import Progress

from utils import GDriveDownloader
from utils import SpatialProjection
from utils import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

# with Progress() as progress:
#     task = progress.add_task(
#         description="processing data ... ",
#         total=(25 * 16)  # n_users * n_gestures
#     )

#     # clean_dir(config["image_dir"])

#     subjects = []
#     labels = np.array([], dtype="<U")
#     images = np.array([], dtype="<U32")
#     data_channels = np.array([], dtype="float32")

#     for u_idx, user in enumerate(config["users"]):
#         for g_idx, gesture in enumerate(config["dynamic_gestures"]):
#             path = os.path.join(
#                 config["data_dir"], user, gesture + ".csv"
#             )
#             data = pd.read_csv(path)
#             channels = preposses_data(data)
#             if data_channels.size == 0:
#                 data_channels = channels
#             else:
#                 data_channels = np.vstack((data_channels, channels))

#             accx = channels[:, :, 5]
#             accy = channels[:, :, 6]
#             accz = channels[:, :, 7]

#             image_prefix = f"{user}_{gesture}"

#             img_files = projection.generate_images(
#                 acceleration=(accx, accy, accz),
#                 image_dir=config["image_dir"],
#                 image_prefix=image_prefix
#             )

#             n = channels.shape[0]
#             subjects += ([user] * n)
#             labels = np.append(labels, [gesture] * n)
#             images = np.append(images, img_files)

#             progress.update(
#                 task_id=task,
#                 description=f"User [{u_idx + 1:>2}/{25}] "
#                 f"Gesture [{g_idx + 1:>2}/{16}] ",
#                 advance=1
#             )


# print(images.shape)

# train_ds, test_ds = load_ds(
#     test_subjects=["001"],
#     data_channels=data_channels,
#     subjects=subjects,
#     images=images,
#     labels=labels,
#     batch_size=2
# )

# print(train_ds.element_spec)
