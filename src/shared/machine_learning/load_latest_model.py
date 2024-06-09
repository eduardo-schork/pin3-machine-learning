import os
import tensorflow as tf


def load_latest_model(model_type, specific_version=None):
    base_dir = "../trained_model"
    model_dir = os.path.join(base_dir, model_type)

    if specific_version:
        version_dir = os.path.join(model_dir, specific_version)
        if not os.path.isdir(version_dir):
            raise ValueError(
                f"Versão {specific_version} não encontrada para {model_type}."
            )
        latest_model_path = version_dir
    else:
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

    print(f"Carregado modelo: {latest_model_path}")

    return loaded_model
