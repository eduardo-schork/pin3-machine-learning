from shared.machine_learning.save_model import save_model
from shared.machine_learning.preprocess_image_set import (
    preprocess_training_set,
    preprocess_validation_set,
)

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def craete_model():
    base_model = VGG16(weights="imagenet", include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(lr=1e-4), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model():
    train_data = preprocess_training_set()
    validation_data = preprocess_validation_set()

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print("Training VGG16 model")
    model_vgg = craete_model()

    model_vgg.fit(
        train_data,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        validation_data=validation_data,
        validation_steps=validation_data.samples // validation_data.batch_size,
        epochs=50,
        callbacks=[reduce_lr, early_stopping],
    )

    save_model(model_vgg, "vgg16")
    print("VGG16 saved")
