import os
import numpy as np
import matplotlib.pyplot as plt

from .dir_utils import clean_dir


class SpatialProjection:

    projection_planes = ["xy", "yz", "zx"]

    def __init__(
            self,
            n_points: int = 150,
            width: int = 100,
            dt: float = 0.01
    ):
        self.n_points = n_points
        self.width = width
        self.dt = dt

    def write_image(
        self,
        x: np.ndarray,
        y: np.ndarray,
        path: os.PathLike
    ):
        plt.subplots(frameon=True, figsize=(3, 3))
        plt.axis('off')
        plt.scatter(x, y, s=self.width, c='black')
        plt.savefig(path)
        plt.close()

    def get_displacement_vector(
        self,
        acceleration: np.ndarray
    ) -> np.ndarray:
        acceleration = acceleration.ravel()
        velocity = np.zeros_like(acceleration)
        displacement = np.zeros_like(acceleration)

        for i in range(acceleration.shape[0] - 1):
            velocity[i + 1] = velocity[i] + acceleration[i] * self.dt
            displacement[i + 1] = velocity[i] * self.dt + \
                0.5 * acceleration[i] * self.dt * self.dt

        return displacement

    def generate_images(
        self,
        acceleration: tuple,
        image_dir: os.PathLike,
        image_prefix: str
    ):
        image_names = []
        x = self.get_displacement_vector(
            acceleration=acceleration[0]
        ).reshape(-1, self.n_points)

        y = self.get_displacement_vector(
            acceleration=acceleration[1]
        ).reshape(-1, self.n_points)

        z = self.get_displacement_vector(
            acceleration=acceleration[2]
        ).reshape(-1, self.n_points)

        for i in range(x.shape[0]):
            for plane in SpatialProjection.projection_planes:
                image_name = f"{image_prefix}_{i:0>3d}_{plane}.jpg"
                image_path = os.path.join(image_dir, image_name)

                # if plane == "xy":
                #     self.write_image(x[i, :], y[i, :], image_path)
                # elif plane == "yz":
                #     self.write_image(y[i, :], z[i, :], image_path)
                # elif plane == "zx":
                #     self.write_image(z[i, :], x[i, :], image_path)
                # else:
                #     pass

                image_names.append(image_name)

        return image_names
