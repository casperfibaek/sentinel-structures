import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from training_utils import get_layer

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "path/to/data"


def test_model(model_path, label, inputs, tile_size="32x32"):
    for place in ["holsterbro", "aarhus", "samsoe"]:
        x_test = get_layer(folder, inputs, prefix=place + "_", tile_size=tile_size)
        y_test = np.load(folder + f"patches/{tile_size}/{place}_label_{label}.npy")

        if label == "people":
            y_test = y_test * 100

        model = tf.keras.models.load_model(model_path)

        _loss, mse, mae = model.evaluate(
            x=x_test,
            y=y_test,
            verbose=1,
            batch_size=32,
            use_multiprocessing=True,
        )

        print(place)
        print(f"Path: {model_path}")
        print(f"Label: {label}")
        print(f"Mean Square Error:      {round(mse, 5)}")
        print(f"Mean Absolute Error:    {round(mae, 5)}")
        print("")


test_model(
    folder + "models/big_model_32x32_people",
    "people",
    "RGBN_RESWIR_VVa_VVd_COHa_COHd",
)
