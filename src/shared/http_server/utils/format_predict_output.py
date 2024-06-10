def format_predict_output(predictions):
    classes = ["peach", "pomegranate", "strawberry"]

    formatted_output = {
        "percentages": {
            classes[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(classes))
        },
        "predictValues": {
            classes[i]: float(predictions[i]) for i in range(len(classes))
        },
        "predominantClass": classes[predictions.argmax()],
    }

    return formatted_output
