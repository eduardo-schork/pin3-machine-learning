from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    )

    return train_generator


def preprocess_test_set():
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        "dataset/test_set",
        target_size=(300, 300),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    return test_generator


def preprocess_validation_set():
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_directory(
        "dataset/validation_set",
        target_size=(300, 300),
        batch_size=32,
        class_mode="categorical",
    )

    return validation_generator
