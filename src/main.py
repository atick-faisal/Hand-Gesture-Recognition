import os
import json

import pandas as pd

from rich.progress import Progress

from utils import GDriveDownloader
from utils import SpatialProjection
from utils import preposses_data
from utils import clean_dir


# ... config
config_file = open("config/config.json")
config = json.load(config_file)
config_file.close()

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

with Progress() as progress:
    task = progress.add_task(
        description="processing data ... ",
        total=(25 * 16)  # n_users * n_gestures
    )

    clean_dir(config["image_dir"])

    subjects = []
    labels = []

    for u_idx, user in enumerate(config["users"]):
        for g_idx, gesture in enumerate(config["dynamic_gestures"]):
            path = os.path.join(
                config["data_dir"], user, gesture + ".csv"
            )
            data = pd.read_csv(path)
            channels = preposses_data(data)

            accx = channels[:, :, 5]
            accy = channels[:, :, 6]
            accz = channels[:, :, 7]

            image_prefix = f"{user}_{gesture}"

            projection.generate_images(
                acceleration=(accx, accy, accz),
                image_dir=config["image_dir"],
                image_prefix=image_prefix
            )

            n = channels.shape[0]
            subjects.append([user] * n)
            labels.append([gesture] * n)

            progress.update(
                task_id=task,
                description=f"User [{u_idx + 1:>2}/{25}] "
                f"Gesture [{g_idx + 1:>2}/{16}] ",
                advance=1
            )

# print(channels.shape)
