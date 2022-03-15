from tensorflow.keras import layers, models


def ConvBlock2D(
    img_size: int,
    base_model: models.Model
):
    preprocess = layers.experimental.preprocessing.Rescaling(
        scale=1.0/127.5,
        offset=-1
    )
    base_model.trainable = False
    global_average = layers.GlobalAveragePooling2D()

    input = layers.Input(shape=(img_size, img_size, 3))
    x = preprocess(input)
    x = base_model(x)
    output = global_average(x)

    return input, output
