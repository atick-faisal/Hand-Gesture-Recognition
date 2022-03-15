from tensorflow.keras import layers, models

from .conv1d import ConvBlock1D
from .conv2d import ConvBlock2D


class ProjectionNet():
    def __init__(
        self,
        img_size: int,
        segment_len: int,
        n_classes: int,
        base_model: models.Model
    ):
        self.img_size = img_size
        self.segment_len = segment_len
        self.base_model = base_model

        self.dropout = layers.Dropout(0.5)

        self.mlp1 = layers.Dense(
            units=128,
            activation="relu"
        )

        self.mlp2 = layers.Dense(
            units=n_classes,
            activation="softmax"
        )

    def get_model(
        self,
        n_projections: int = 3,
        n_channels: int = 5
    ):

        inputs = []
        features = []

        for _ in range(n_projections):
            input_2d, features_2d = ConvBlock2D(
                img_size=self.img_size,
                base_model=self.base_model
            )

            inputs.append(input_2d)
            features.append(features_2d)

        for _ in range(n_channels):
            input_1d, features_1d = ConvBlock1D(
                segment_len=self.segment_len
            )

            inputs.append(input_1d)
            features.append(features_1d)

        x = layers.concatenate(features, axis=-1)
        x = self.dropout(x)
        x = self.mlp1(x)
        x = self.dropout(x)
        output = self.mlp2(x)

        return models.Model(inputs, output)
