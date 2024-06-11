from shared.machine_learning.preprocess_image_set import (
    preprocess_image_set_convnet,
    preprocess_validation_set_convnet,
)
from shared.machine_learning.save_model import save_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)


# DEPRECIATED
def create_model(num_classes):
    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(300, 300, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=126, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.15))
    model.add(Flatten())
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(rate=0.15))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=num_classes, activation="softmax"))
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_convnet_model():
    num_classes = 3
    model_instance = create_model(num_classes)

    training_set = preprocess_image_set_convnet("dataset/training_set")
    validation_set = preprocess_validation_set_convnet("dataset/validation_set")

    print("Treinando o modelo convnet...")
    model_instance.fit(
        training_set,
        epochs=25,
        validation_data=validation_set,
    )

    # Salvar o modelo treinado
    save_model(model_instance, "convnet")
