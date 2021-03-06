import os
import numpy as np
import matplotlib.pyplot as plt


class SpatialProjection:

    projection_planes = ["xy", "yz", "zx"]

    def __init__(
            self,
            width: int = 100,
            dt: float = 0.01
    ):
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
        image_dir: str,
        image_prefix: str
    ):
        image_names = []
        x = self.get_displacement_vector(acceleration=acceleration[0])
        y = self.get_displacement_vector(acceleration=acceleration[1])
        z = self.get_displacement_vector(acceleration=acceleration[2])

        for plane in SpatialProjection.projection_planes:
            image_name = f"{image_prefix}_{plane}.jpg"
            image_path = os.path.join(image_dir, image_name)

            if plane == "xy":
                self.write_image(x, y, image_path)
            elif plane == "yz":
                self.write_image(y, z, image_path)
            elif plane == "zx":
                self.write_image(z, x, image_path)
            else:
                pass

            image_names.append(
                os.path.join(image_dir, image_name)
            )

        return image_names
