from tensorflow.keras import layers, models


class ConvBlock2D(models.Model):
    def __init__(
        self,
        img_size: int,
        base_model: models.Model
    ):
        super().__init__()
        self.preprocess = layers.experimental.preprocessing.Rescaling(
            scale=1.0/127.5,
            offset=-1
        )
        self.base_model = base_model
        self.base_model.trainable = False
        self.global_average = layers.GlobalAveragePooling2D()

    def call(self, inputs):
        x = self.preprocess(inputs)
        x = self.base_model(x)
        output = self.global_average(x)

        return models.Model(inputs, output)
