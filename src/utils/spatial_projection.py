import os
import numpy as np
import matplotlib.pyplot as plt

from rich.status import Status
from .dir_utils import clean_dir

status = Status("", spinner="dots9")
# status.start()


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
        image_dir: os.PathLike
    ):
        x = self.get_displacement_vector(
            acceleration=acceleration[0]
        ).reshape(-1, self.n_points)

        y = self.get_displacement_vector(
            acceleration=acceleration[1]
        ).reshape(-1, self.n_points)

        z = self.get_displacement_vector(
            acceleration=acceleration[2]
        ).reshape(-1, self.n_points)

        for plane in SpatialProjection.projection_planes:
            for i in range(x.shape[0]):
                image_name = "{:0>3d}".format(i) + ".jpg"
                path = os.path.join(image_dir, image_name)

                if plane == "xy":
                    self.write_image(x[i, :], y[i, :], path)
                elif plane == "yz":
                    self.write_image(y[i, :], z[i, :], path)
                elif plane == "zx":
                    self.write_image(z[i, :], x[i, :], path)
                else:
                    pass
