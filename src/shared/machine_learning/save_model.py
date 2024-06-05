import os
from datetime import datetime


def save_model(model, model_type):
    base_dir = "../trained_model"

    model_dir = os.path.join(base_dir, model_type)

    existing_versions = [
        int(d.split("v")[-1])
        for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ]
    latest_version = max(existing_versions, default=0)

    new_version = latest_version + 1
    version_dir = os.path.join(model_dir, f"v{new_version}")
    os.makedirs(version_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{timestamp}_model.h5"
    model_path = os.path.join(version_dir, model_filename)

    model.save(model_path)
