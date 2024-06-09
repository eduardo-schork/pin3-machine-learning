from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


def plot_processed_images(data_generator):
    num_cols = 10

    for i in range(len(data_generator)):
        x_batch, y_batch = next(data_generator)

        num_images = len(x_batch)
        num_rows = (num_images + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
        axes = axes.flatten()
        for img, ax in zip(x_batch, axes):
            ax.imshow(img)
            ax.axis("off")

        for ax in axes[num_images:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def preprocess_training_set():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_generator = train_datagen.flow_from_directory(
        "dataset/training_set",
        target_size=(300, 300),
        batch_size=32,
        class_mode="categorical",
        color_mode="rgb",
    )

    # plot_processed_images(train_generator)

    print("Training classes:", train_generator.class_indices)
    return train_generator


def preprocess_test_set():
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        "dataset/test_set",
        target_size=(300, 300),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
        color_mode="rgb",
    )

    # plot_processed_images(test_generator)

    print("Test classes:", test_generator.class_indices)
    return test_generator


def preprocess_validation_set():
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_directory(
        "dataset/validation_set",
        target_size=(300, 300),
        batch_size=32,
        class_mode="categorical",
        color_mode="rgb",
    )

    # plot_processed_images(validation_generator)

    print("Validation classes:", validation_generator.class_indices)
    return validation_generator
