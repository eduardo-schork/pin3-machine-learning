from shared.machine_learning.models.inceptionv3_model import train_inceptionv3_model
from shared.machine_learning.models.vgg16_model import train_vgg16_model

# from shared.machine_learning.models.convnet_model import train_convnet_model


train_vgg16_model()
train_inceptionv3_model()
# train_convnet_model()

print("Treinamento concluído para ambos os modelos.")
