import traceback
from tensorflow.keras.preprocessing.image import img_to_array
from ..utils.validate_image_input import validate_image_input


def preprocess_image(img):
    try:
        is_valid, img = validate_image_input(img)
        if not is_valid:
            return None

        img = img.resize((300, 300))
        img_array = img_to_array(img)
        img_array = img_array.reshape(
            (1, img_array.shape[0], img_array.shape[1], img_array.shape[2])
        )
        img_array = img_array / 255.0

        return img_array
    except Exception as e:
        traceback.print_exc()
        return None
