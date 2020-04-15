# GPUU Bugfix
import tensorflow

physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from awesome_callback import MyCallback
from dataset_combinator import tf_datasets_from_dir

total_classes, train_data, val_data, test_data = tf_datasets_from_dir()
print(train_data)
print(test_data)


def make_model():
    layers = [
        Conv2D(64, 5, input_shape=(200, 60, 3)),
        Conv2D(16, 3),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(total_classes, activation="softmax"),
    ]
    model = Sequential(layers)
    model.compile(Adam(), SparseCategoricalCrossentropy())
    return model


model = make_model()
model.fit(
    train_data,
    epochs=128,
    validation_data=val_data,
    callbacks=[MyCallback("testrun", 5)],
)
