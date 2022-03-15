from tensorflow.keras import layers, models


def ConvBlock1D(segment_len: int):
    cnn1 = layers.Conv1D(
        filters=8,
        kernel_size=3,
        activation="relu",
        padding="valid"
    )

    cnn2 = layers.Conv1D(
        filters=16,
        kernel_size=3,
        activation="relu",
        padding="valid"
    )

    pool = layers.MaxPool1D(2)

    flatten = layers.Flatten()

    mlp = layers.Dense(
        units=50,
        activation="relu"
    )

    input = layers.Input(shape=(segment_len, 1))
    x = cnn1(input)
    x = pool(x)
    x = cnn2(x)
    x = pool(x)
    x = flatten(x)
    output = mlp(x)

    return input, output
