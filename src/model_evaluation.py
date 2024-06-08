import numpy as np
from shared.machine_learning.load_latest_model import load_latest_model
from shared.machine_learning.preprocess_image_set import preprocess_image_set


def evaluate_model(model, test_set):
    predictions = model.predict(
        test_set, steps=test_set.n // test_set.batch_size + 1, verbose=1
    )
    y_true = test_set.classes
    y_pred = np.argmax(predictions, axis=1)

    print(y_true)
    print(y_pred)

    accuracy = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    f1_score = tf.keras.metrics.Mean()

    accuracy.update_state(y_true, y_pred)
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)

    f1_score.update_state(
        (
            2
            * (precision.result() * recall.result())
            / (precision.result() + recall.result())
        )
    )

    print("Accuracy: ", accuracy.result().numpy())
    print("Precision: ", precision.result().numpy())
    print("Recall: ", recall.result().numpy())
    print("F1 Score: ", f1_score.result().numpy())


import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    test_dir = "dataset/test_set"

    model_vgg16 = load_latest_model("vgg16")
    model_inceptionv3 = load_latest_model("inceptionv3")

    print("Avaliação do modelo InceptionV3:")
    evaluate_model(model_inceptionv3, preprocess_image_set(test_dir))

    print("Avaliação do modelo VGG16:")
    evaluate_model(model_vgg16, preprocess_image_set(test_dir))
