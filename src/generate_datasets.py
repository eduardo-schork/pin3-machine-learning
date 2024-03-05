import os
import shutil
import random

training_set_folder_path = "../src/dataset/training_set"
validation_set_folder_path = "../src/dataset/validation_set"
test_set_folder_path = "../src/dataset/test_set"


fruits_list = ["strawberry", "peach", "pomegranate"]

for fruit in fruits_list:
    os.makedirs(f"../assets/{fruit}", exist_ok=True)
    os.makedirs(f"{training_set_folder_path}/{fruit}", exist_ok=True)
    os.makedirs(f"{validation_set_folder_path}/{fruit}", exist_ok=True)
    os.makedirs(f"{test_set_folder_path}/{fruit}", exist_ok=True)


def copy_images(fruit, training_ratio=0.7, validation_ratio=0.15):
    images_path = f"../assets/{fruit}/"
    training_set_path = f"{training_set_folder_path}/{fruit}/"
    validation_set_path = f"{validation_set_folder_path}/{fruit}/"
    test_set_path = f"{test_set_folder_path}/{fruit}/"

    all_images = os.listdir(images_path)
    random.shuffle(all_images)

    total_images = len(all_images)
    total_training = int(total_images * training_ratio)
    total_validation = int(total_images * validation_ratio)
    # total_test = total_images - total_training - total_validation

    for i, image_name in enumerate(all_images):
        source_path = os.path.join(images_path, image_name)

        if i < total_training:
            destination_path = training_set_path
        elif i < total_training + total_validation:
            destination_path = validation_set_path
        else:
            destination_path = test_set_path

        shutil.copy(source_path, destination_path)


for fruit in fruits_list:
    copy_images(fruit, training_ratio=0.7, validation_ratio=0.15)
