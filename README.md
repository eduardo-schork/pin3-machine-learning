
# Grupo Flora
## Resumo
Este projeto visa criar um sistema de classificação de frutas usando modelos de Machine Learning (ML). Os modelos treinados são disponibilizados através de uma API HTTP que permite a previsão da classe de uma fruta com base em uma imagem fornecida.

### Algoritmos de Machine Learning
O projeto utiliza os seguintes algoritmos de ML:

**InceptionV3**: Arquitetura de rede neural profunda pré-treinada para classificação de imagens.

**VGG16**: Modelo de rede neural pré-treinado convolucional para classificação de imagens.

### Documentação da API
A API consiste em três endpoints principais:

**/predict**: Realiza previsões utilizando todos os modelos.

**/predict/vgg16**: Realiza previsões utilizando o modelo VGG16.

**/predict/inceptionv3**: Realiza previsões utilizando o modelo InceptionV3.

### Instalação do Python (Windows e macOS)
Antes de começar, certifique-se de ter o Python instalado em seu sistema. Siga as instruções abaixo com base no seu sistema operacional.

#### Windows
1. Baixe o instalador do Python do site oficial.
2. Execute o instalador e marque a opção "Adicionar o Python 3.x ao PATH" durante a instalação.
3. Clique em "Install Now" e aguarde a conclusão da instalação.

#### macOS
1. Abra o terminal.
2. Instale o gerenciador de pacotes Homebrew usando o seguinte comando:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

3. Após a instalação do Homebrew, instale o Python usando o seguinte comando:
```bash
brew install python
```

### Setup de Desenvolvimento e Instalação de Dependências
**No seu terminal**

Clone o Repositório:


```bash
git clone https://github.com/eduardo-schork/pin3-machine-learning

cd pin3-machine-learning
```

Crie a estrutura de pastas necessária para o projeto:
```bash
# Para Linux/Mac
# na pasta raiz do projeto `pin3-machine-learning`

chmod +x ./create_folders.sh

./create_folders.sh

```
Crie e Ative um Ambiente Virtual (venv):

```bash
pip install virtualenv

python -m venv venv

source venv/bin/activate  # Para Linux/Mac
.\venv\Scripts\activate   # Para Windows
```

Instale as Dependências:

```bash
pip install -r requirements.txt
```

### Rodando os Scripts

#### todos os scripts python dever ser rodados de dentro da pasta `/src` do projeto

1. generate_datasets.py

Este script gera conjuntos de dados para treinamento, validação e testes a partir de imagens de frutas. Certifique-se de organizar as imagens corretamente antes de executar o script. 

O Script buscará as imagens (fornecidas pelo professor) na raiz do projeto dentro de `assets`.
```
Comando para rodar o script:

```bash
python generate_datasets.py
```

2. train_models.py

Este script treina modelos de ML utilizando diferentes algoritmos para classificar frutas. Os modelos treinados são salvos para uso posterior.

Comando para rodar o script:

```bash
python train_models.py
```

3. model_evaluation.py

Este script avalia o desempenho dos modelos treinados mais recentes com base em conjuntos de dados de teste.

Comando para rodar o script:

```bash
python model_evaluation.py
```

4. run_server.py

Este script inicia um servidor HTTP que permite realizar previsões de classificação de frutas através de uma API.

Comando para rodar o script:

```bash
python run_server.py
```

### Estrutura do Projeto

```bash
./src
├── shared
│   ├── http_server
|   |   ├── controllers
│   │   │   ├── auth_controller.py
│   │   │   ├── post_controller.py
│   │   │   └── predict_controller.py
|   |   ├── services
│   │   │   ├── firebase_service.py
│   │   │   └── image_service.py
|   |   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── validate_image_input.py
│   │   │   └── format_predict_output.py
│   │   ├── templates
│   │   │   └── index.html
│   │   ├── http_server.py
│   └── machine_learning
│       ├── models
│       │   ├── inceptionv3_model.py
│       │   └── vgg16_model.py
│       ├── preprocess_image_set.py
│       ├── save_model.py
│       └── load_latest_model.py
├── generate_datasets.py
├── model_evaluation.py
├── run_server.py
├── train_model_inception.py
└── train_model_vgg.py
```
* **generate_datasets.py**: Gera conjuntos de dados.
* **train_models.py**: Treina modelos de ML.
* **model_evaluation.py**: Avalia o desempenho dos modelos.
* **run_server.py**: Inicia o servidor HTTP.
* **shared**: Módulo compartilhado.
    * **dataset**: Pasta contendo os datasets de treinamento, validação e testes gerados pelo script `generate_datasets.py`.
    * **http_server**: Módulo para o servidor HTTP.
    * **machine_learning**: Módulo para o treinamento e avaliação de modelos de ML.

