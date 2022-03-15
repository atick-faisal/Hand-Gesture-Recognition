import os
import json
import tensorflow as tf

from rich.progress import Progress
from tensorflow.keras import applications, losses, optimizers

from model import ProjectionNet
from utils import DataLoader
from utils import SpatialProjection
from utils import GDriveDownloader


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

EXPERIMENT = "DYNAMIC"  # STATIC or DYNAMIC


# ... config
config_file = open("config/config.json")
config = json.load(config_file)
config_file.close()

gestures = None
if EXPERIMENT == "STATIC":
    gestures = config["static_gestures"]
elif EXPERIMENT == "DYNAMIC":
    gestures = config["dynamic_gestures"]
else:
    raise ValueError("Wrong experiment!")

projection = SpatialProjection(
    width=config["line_width"],
    dt=config["dt"]
)

# ... Download files
downloader = GDriveDownloader()
# downloader.download(
#     fid=config["dataset_id"],
#     destination=config["data_dir"]
# )

dataloader = DataLoader(
    data_dir=config["data_dir"],
    images_dir=config["images_dir"],
    channels_dir=config["channels_dir"],
    users=config["users"],
    gestures=gestures
)

dataloader.extract_channels(
    fs=config["fs"],
    imu_cutoff=config["imu_cutoff"],
    window_len=config["segment_len"]
)

# dataloader.generate_projection_images(projection)

base_model = applications.MobileNetV2(
    input_shape=(config["img_size"], config["img_size"], 3),
    include_top=False,
    weights='imagenet'
)

projection_net = ProjectionNet(
    img_size=config["img_size"],
    segment_len=config["segment_len"],
    n_classes=len(gestures),
    base_model=base_model
)

model = projection_net.get_model(
    n_projections=3,
    n_channels=5
)

loss = losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=["accuracy"]
)

model.summary()
model.save_weights(
    os.path.join(config["models_dir"], "initial_weights")
)

for test_user in config["users"]:
    print("===========================================")
    print(f"                  {test_user}")
    print("===========================================")
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=config["logs_dir"] + test_user,
        histogram_freq=1,
        update_freq=1
    )
    train_ds, test_ds = dataloader.load_ds(
        test_subjects=[test_user],
        image_shape=(config["img_size"], config["img_size"]),
        batch_size=config["batch_size"]
    )
    model.load_weights(
        os.path.join(config["models_dir"], "initial_weights")
    )

    model.fit(
        train_ds,
        batch_size=config["batch_size"],
        epochs=config["n_epochs"],
        callbacks=[tb_callback]
    )

    model.evaluate(test_ds)
    model.save_weights(
        os.path.join(config["models_dir"], test_user)
    )

    tf.keras.backend.clear_session()
