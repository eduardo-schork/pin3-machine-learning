from shared.machine_learning.preprocess_image_set import preprocess_image_set
from shared.machine_learning.save_model import save_model

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers.legacy import Adam


def create_model(num_classes):
    base_model = InceptionV3(
        weights="imagenet", include_top=False, input_shape=(300, 300, 3)
    )

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
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


def train_inceptionv3_model():
    num_classes = 3

    inceptionv3_model = create_model(num_classes)

    training_set = preprocess_image_set("dataset/training_set")
    validation_set = preprocess_image_set("dataset/validation_set")

    print("Treinando o modelo InceptionV3...")
    inceptionv3_model.fit(
        training_set,
        epochs=25,
        validation_data=validation_set,
    )

    save_model(inceptionv3_model, "inceptionv3")
