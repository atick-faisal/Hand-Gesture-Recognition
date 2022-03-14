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
            gestures: list
    ):
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.channels_dir = channels_dir
        self.users = users
        self.gestures = gestures

    def extract_channels(
        self,
        fs: int,
        imu_cutoff: int,
        window_len: int
    ):
        if not os.path.exists(self.channels_dir):
            os.makedirs(self.channels_dir)

        data_channels = np.array([], dtype="float64")
        labels = np.array([], dtype="uint8")
        subjects = np.array([], dtype="<U3")

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
                    channels = preposses_data(
                        data=data,
                        fs=fs,
                        imu_cutoff=imu_cutoff,
                        window_len=window_len
                    )
                    if data_channels.size == 0:
                        data_channels = channels
                    else:
                        data_channels = np.vstack(
                            (data_channels, channels)
                        )
                    n = channels.shape[0]
                    labels = np.append(labels, [g_idx] * n)
                    subjects = np.append(subjects, [user] * n)

                    progress.update(
                        task_id=task,
                        description=f"User [{u_idx + 1:>2}/{25}] "
                        f"Gesture [{g_idx + 1:>2}/{16}] ",
                        advance=1
                    )

            np.save(os.path.join(
                self.channels_dir, "channels.npy"
            ), data_channels)

            np.save(os.path.join(
                self.channels_dir, "labels.npy"
            ), labels)

            np.save(os.path.join(
                self.channels_dir, "subjects.npy"
            ), subjects)

    def generate_projection_images(
        self,
        projection: SpatialProjection
    ):
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        data_channels = np.load(
            os.path.join(
                self.channels_dir, "channels.npy"
            )
        )

        accx = data_channels[:, :, 5]
        accy = data_channels[:, :, 6]
        accz = data_channels[:, :, 7]

        images = np.array([], dtype="<U16")

        with Progress() as progress:
            task = progress.add_task(
                description="processing data ... ",
                total=(len(self.users) * len(self.gestures))
            )

            for u_idx, user in enumerate(self.users):
                for g_idx, _ in enumerate(self.gestures):
                    image_prefix = f"U{user}_{g_idx:0>3}"
                    img_files = projection.generate_images(
                        acceleration=(accx, accy, accz),
                        image_dir=self.images_dir,
                        image_prefix=image_prefix
                    )

                    images = np.append(images, img_files)

                    progress.update(
                        task_id=task,
                        description=f"User [{u_idx + 1:>2}/{25}] "
                        f"Gesture [{g_idx + 1:>2}/{16}] ",
                        advance=1
                    )

        np.save(os.path.join(
            self.images_dir, "image_names.npy"
        ), images)

    @staticmethod
    def load(
        file_path: str,
        img_shape: tuple
    ) -> tf.Tensor:
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, size=img_shape)

        return img

    @staticmethod
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
        self,
        test_subjects: list,
        image_shape: tuple,
        batch_size: int
    ):
        data_channels = np.load(
            os.path.join(
                self.channels_dir, "channels.npy"
            )
        )

        labels = np.load(
            os.path.join(
                self.channels_dir, "labels.npy"
            )
        )

        subjects = np.load(
            os.path.join(
                self.channels_dir, "subjects.npy"
            )
        )

        images = np.load(
            os.path.join(
                self.images_dir, "image_names.npy"
            )
        )

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
            lambda img_xy, img_yz, img_zx, *channels:
            (
                DataLoader.load(img_xy, image_shape),
                DataLoader.load(img_yz, image_shape),
                DataLoader.load(img_zx, image_shape),
                channels
            )
        )

        test_data = tf.data.Dataset.from_tensor_slices(
            (test_images_xy, test_images_yz, test_images_zx,
             *np.split(test_channels, test_channels.shape[-1], axis=-1))
        ).map(
            lambda img_xy, img_yz, img_zx, *channels:
            (
                DataLoader.load(img_xy, image_shape),
                DataLoader.load(img_yz, image_shape),
                DataLoader.load(img_zx, image_shape),
                channels
            )
        )

        train_labels = tf.data.Dataset.from_tensor_slices(labels[~mask])
        test_labels = tf.data.Dataset.from_tensor_slices(labels[mask])

        train_ds = tf.data.Dataset.zip((train_data, train_labels))
        test_ds = tf.data.Dataset.zip((test_data, test_labels))

        train_ds = DataLoader.configure_for_performance(
            train_ds,
            batch_size
        )
        test_ds = DataLoader.configure_for_performance(
            test_ds,
            batch_size
        )

        return train_ds, test_ds
