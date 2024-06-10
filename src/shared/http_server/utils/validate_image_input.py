from PIL import Image
from io import BytesIO


def validate_image_input(file):
    if file is None:
        return False, "Nenhum arquivo de imagem fornecido."

    try:
        img = Image.open(BytesIO(file.stream.read()))
        img = img.convert("RGB")
        img = img.resize((300, 300))
        return True, img
    except Exception as e:
        return False, f"Erro ao processar a imagem: {str(e)}"
