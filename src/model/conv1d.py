from tensorflow.keras import layers, models


class ConvBlock1D(models.Model):
    def __init__(
        self,
        segment_len: int
    ):
        super().__init__()
        self.input_ = layers.Input(shape=(segment_len, 1))
        self.cnn1 = layers.Conv1D(
            filters=8,
            kernel_size=3,
            activation="relu",
            padding="valid"
        )
        self.cnn2 = layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation="relu",
            padding="valid"
        )
        self.pool = layers.MaxPool1D(2)
        self.flatten = layers.Flatten()
        self.mlp = layers.Dense(
            units=50,
            activation="relu"
        )

    def call(self, x):
        x = self.input_(x)
        x = self.cnn1(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.mlp(x)

        return x
