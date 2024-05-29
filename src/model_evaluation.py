import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from shared.machine_learning.load_latest_model import load_latest_model
from shared.machine_learning.preprocess_image_set import (
    preprocess_image_set,
    preprocess_validation_set_convnet,
)


def evaluate_model(model, test_generator):
    evaluation = model.evaluate(test_generator, verbose=1)

    predictions = model.predict(test_generator)

    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_generator.classes

    class_names = list(test_generator.class_indices.keys())
    print(
        "Relatório de Classificação:\n",
        classification_report(
            true_classes, predicted_classes, target_names=class_names
        ),
    )

    cm = confusion_matrix(true_classes, predicted_classes)
    print("Matriz de Confusão:\n", cm)

    plot_confusion_matrix(cm, class_names)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")
    plt.show()


if __name__ == "__main__":
    # Diretórios dos conjuntos de teste
    test_dir = "/Users/eduardo-schork/workspace/pin3-ml/src/dataset/test_set"

    model_vgg16 = load_latest_model("vgg16")
    model_inceptionv3 = load_latest_model("inceptionv3")
    model_convnet = load_latest_model("convnet")

    test_set = preprocess_image_set(test_dir)

    print("\nAvaliação do Modelo VGG16:")
    evaluate_model(model_vgg16, test_set)

    print("\nAvaliação do Modelo InceptionV3:")
    evaluate_model(model_inceptionv3, test_set)

    print("\nAvaliação do Modelo convnet:")
    convnet_test_set = preprocess_validation_set_convnet(test_dir)
    evaluate_model(model_convnet, convnet_test_set)
