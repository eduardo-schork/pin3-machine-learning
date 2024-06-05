import os

import tensorflow as tf


def load_latest_model(model_type):
    base_dir = "../trained_model"

    model_dir = os.path.join(base_dir, model_type)

    existing_versions = [
        int(d.split("v")[-1])
        for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ]
    latest_version = max(existing_versions, default=0)

    if latest_version == 0:
        raise ValueError("Nenhum modelo treinado encontrado.")

    latest_model_path = os.path.join(model_dir, f"v{latest_version}")

    model_files = [f for f in os.listdir(latest_model_path) if f.endswith("_model.h5")]

    if not model_files:
        raise ValueError(
            f"Nenhum arquivo de modelo (.h5) encontrado em {latest_model_path}"
        )

    latest_model_file = model_files[0]
    latest_model_path = os.path.join(latest_model_path, latest_model_file)

    loaded_model = tf.keras.models.load_model(latest_model_path)

    print(latest_model_path)

    return loaded_model
