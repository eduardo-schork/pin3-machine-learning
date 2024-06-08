from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from shared.machine_learning.load_latest_model import load_latest_model
from shared.machine_learning.preprocess_image_set import preprocess_test_set


def evaluate_model(model_type):
    test_data = preprocess_test_set()

    model = load_latest_model(model_type)
    Y_pred = model.predict(test_data)

    print(Y_pred)

    y_pred = np.argmax(Y_pred, axis=1)
    print("Confusion Matrix")

    print(confusion_matrix(test_data.classes, y_pred))
    print("Classification Report")

    target_names = ["Peach", "Strawberry", "Pomegranate"]

    print(classification_report(test_data.classes, y_pred, target_names=target_names))


print("Evaluating model Inceptionv3")
evaluate_model("inceptionv3")

print("Evaluating model Vgg16")
evaluate_model("vgg16")
