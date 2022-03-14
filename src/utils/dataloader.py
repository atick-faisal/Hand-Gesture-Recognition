import os
import numpy as np
import pandas as pd
import tensorflow as tf

from rich.progress import Progress

from .preprocess import preposses_data
from .spatial_projection import SpatialProjection

AUTOTUNE = tf.data.AUTOTUNE


class DataLoader:
    def __init__(
            self,
            data_dir: str,
            images_dir: str,
            channels_dir: str,
            users: list,
            gestures: list,
            projection_generator: SpatialProjection
    ):
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.channels_dir = channels_dir
        self.users = users
        self.gestures = gestures
        self.projection_generator = projection_generator

    def extract_channels(self):
        if not os.path.exists(self.channels_dir):
            os.makedirs(self.channels_dir)

        data_channels = np.array([], dtype="float64")
        labels = np.array([], dtype="uint8")

        with Progress() as progress:
            task = progress.add_task(
                description="processing data ... ",
                total=(len(self.users) * len(self.gestures))
            )

            for u_idx, user in enumerate(self.users):
                for g_idx, gesture in enumerate(self.gestures):
                    path = os.path.join(
                        self.data_dir, user, gesture + ".csv"
                    )
                    data = pd.read_csv(path)
                    channels = preposses_data(data)
                    if data_channels.size == 0:
                        data_channels = channels
                    else:
                        data_channels = np.vstack(
                            (data_channels, channels)
                        )
                    n = channels.shape[0]
                    labels = np.append(labels, [g_idx] * n)

                    progress.update(
                        task_id=task,
                        description=f"User [{u_idx + 1:>2}/{25}] "
                        f"Gesture [{g_idx + 1:>2}/{16}] ",
                        advance=1
                    )

            print(data_channels.shape)
            print(labels.shape)


def load(file_path: str) -> tf.Tensor:
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=(32, 32))
    return img


def configure_for_performance(
    ds: tf.data.Dataset,
    batch_size: int
) -> tf.data.Dataset:
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def load_ds(
    test_subjects: list,
    data_channels: np.ndarray = None,
    subjects: list = None,
    labels: list = None,
    images: list = None,
    batch_size: int = 2
):
    mask = np.array([], dtype="bool")
    for subject in subjects:
        mask = np.append(mask, (subject in test_subjects))

    train_channels = data_channels[~mask, :, :]
    test_channels = data_channels[mask, :, :]

    images_xy = images[0::3]
    images_yz = images[1::3]
    images_zx = images[2::3]

    train_images_xy = images_xy[~mask]
    train_images_yz = images_yz[~mask]
    train_images_zx = images_zx[~mask]

    test_images_xy = images_xy[mask]
    test_images_yz = images_yz[mask]
    test_images_zx = images_zx[mask]

    train_data = tf.data.Dataset.from_tensor_slices(
        (train_images_xy, train_images_yz, train_images_zx,
         *np.split(train_channels, train_channels.shape[-1], axis=-1))
    ).map(
        lambda xy, yz, zx, *channels:
        (load(xy), load(yz), load(zx), channels)
    )

    test_data = tf.data.Dataset.from_tensor_slices(
        (test_images_xy, test_images_yz, test_images_zx,
         *np.split(test_channels, test_channels.shape[-1], axis=-1))
    ).map(
        lambda xy, yz, zx, *channels:
        (load(xy), load(yz), load(zx), channels)
    )

    train_labels = tf.data.Dataset.from_tensor_slices(labels[~mask])
    test_labels = tf.data.Dataset.from_tensor_slices(labels[mask])

    train_ds = tf.data.Dataset.zip((train_data, train_labels))
    test_ds = tf.data.Dataset.zip((test_data, test_labels))

    train_ds = configure_for_performance(train_ds, batch_size)
    test_ds = configure_for_performance(test_ds, batch_size)

    return train_ds, test_ds
