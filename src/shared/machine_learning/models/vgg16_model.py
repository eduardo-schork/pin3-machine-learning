from shared.machine_learning.preprocess_image_set import preprocess_image_set
from shared.machine_learning.save_model import save_model

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam


def create_model(num_classes):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(300, 300, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()

    model.add(base_model)

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    adam_optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model.compile(
        optimizer=adam_optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_vgg16_model():
    num_classes = 3

    vgg16_model = create_model(num_classes)

    training_set = preprocess_image_set("dataset/training_set")
    validation_set = preprocess_image_set("dataset/validation_set")

    print("Treinando o modelo VGG16...")
    vgg16_model.fit(
        training_set,
        epochs=25,
        validation_data=validation_set,
    )

    save_model(vgg16_model, "vgg16")
