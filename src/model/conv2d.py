from tensorflow.keras import layers, models


class ConvBlock2D(models.Model):
    def __init__(
        self,
        img_size: int,
        base_model: models.Model
    ):
        super().__init__()
        self.input_ = layers.Input(shape=(img_size, img_size))
        self.preprocess = layers.experimental.preprocessing.Rescaling(
            scale=1.0/127.5,
            offset=-1
        )
        self.base_model = base_model
        self.base_model.trainable = False
        self.global_average = layers.GlobalAveragePooling2D()

    def call(self, x):
        x = self.input_(x)
        x = self.preprocess(x)
        x = self.base_model(x)
        x = self.global_average(x)

        return x
