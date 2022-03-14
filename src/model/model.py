from tensorflow.keras import layers, models

from .conv1d import ConvBlock1D
from .conv2d import ConvBlock2D


class ProjectionNet(models.Model):
    def __init__(
        self,
        img_size: int,
        segment_len: int,
        n_classes: int,
        base_model: models.Model,
        n_projections: int = 3,
        n_channels: int = 5
    ):
        super().__init__()
        self.n_projections = n_projections
        self.n_channels = n_channels

        self.conv_block_2d = ConvBlock2D(
            img_size=img_size,
            base_model=base_model
        )

        self.conv_block_1d = ConvBlock1D(
            segment_len=segment_len
        )

        self.dropout = layers.Dropout(0.5)

        self.mlp1 = layers.Dense(
            units=128,
            activation="relu"
        )

        self.mlp2 = layers.Dense(
            units=n_classes,
            activation="softmax"
        )

    def call(self, inputs):
        print(len(inputs[0]))
        images = inputs[:self.n_projections]
        channels = inputs[self.n_projections:]

        features = []

        for image in images:
            features.append(
                self.conv_block_2d(image)
            )

        for channel in channels:
            features.append(
                self.conv_block_1d(channel)
            )

        x = layers.concatenate(features, axis=-1)
        x = self.dropout(x)
        x = self.mlp1(x)
        x = self.dropout(x)
        x = self.mlp2(x)

        return x
