import traceback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

classes_order = ["strawberry", "peach", "pomegranate"]

# FIX preprocess_image_set_convnet and preprocess_validation_set_convnet methods,
# isolate and reuse logic


def preprocess_image_set_convnet(path, validation_split=0.2, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    data_set = datagen.flow_from_directory(
        path,
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode="sparse",
        classes=classes_order,
        shuffle=True,
        seed=42,
        interpolation="nearest",
        follow_links=True,
        subset="training",
    )

    return data_set


def preprocess_validation_set_convnet(path, validation_split=0.2, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    data_set = datagen.flow_from_directory(
        path,
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode="sparse",
        classes=classes_order,
        shuffle=True,
        seed=42,
        interpolation="nearest",
        follow_links=True,
        subset="validation",
    )

    return data_set


def preprocess_image_set(path):
    try:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode="nearest",
        )

        data_set = datagen.flow_from_directory(
            path,
            target_size=(300, 300),
            batch_size=32,
            class_mode="categorical",
            classes=classes_order,
        )

        return data_set
    except Exception as e:
        traceback.print_exc()
