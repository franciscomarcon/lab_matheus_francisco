
from pathlib import Path
import torchvision.models as tv_models
import torch.nn as nn
from fastai.vision.all import *


# 1. Diretório raiz do dataset
path = Path('./tiny-imagenet-200')
print("Diretório do dataset:", path.absolute())


# 2. Rótulo da classe (pasta avó: train/<classe_id>/images/<arquivo>.JPEG)
def get_tiny_label(file_path):
   return file_path.parent.parent.name

# 3. DataBlock utilizando apenas o conjunto de treino com divisão interna (ex: 15% validação)
tiny_datablock = DataBlock(
   blocks=(ImageBlock, CategoryBlock),
   get_items=get_image_files,
   splitter=RandomSplitter(valid_pct=0.15, seed=42),
   get_y=get_tiny_label,
   item_tfms=Resize(64),
   batch_tfms=aug_transforms(mult=1.0, do_flip=True, max_rotate=10.0)
)

# 4. Carrega apontando diretamente para a pasta 'train'
dls = tiny_datablock.dataloaders(path / 'train', bs=128, num_workers=4)


print("DataLoaders criados com sucesso!")
print(f"Total de classes: {len(dls.vocab)}")
print(f"Imagens de Treino: {len(dls.train_ds)} | Validação: {len(dls.valid_ds)}")

# 5. SqueezeNet (versão 1_1) com pesos pré-treinados
base_model = tv_models.squeezenet1_1(weights=tv_models.SqueezeNet1_1_Weights.DEFAULT)

# O SqueezeNet usa uma Conv2d de 1x1 na última camada do classificador (em vez de uma Linear padrão).
# Precisamos ajustar o número de canais de saída para 200 (número de classes do Tiny ImageNet).
num_classes = len(dls.vocab)

# A estrutura padrão do classificador do SqueezeNet é: Sequential(Dropout, Conv2d, ReLU, AdaptiveAvgPool2d)
# O índice 1 é a Conv2d que define as classes de saída.
base_model.classifier[1] = nn.Conv2d(
    in_channels=512, 
    out_channels=num_classes, 
    kernel_size=1
)


# 6. Criação do Learner
learn = Learner(
   dls,
   base_model,
   metrics=[accuracy, top_k_accuracy],
   loss_func=CrossEntropyLossFlat()
)

import wandb
wandb.login(key='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') #necessario colocar chave de api ao dar o login, costuma falhar ao colocar a chave de api no terminal do vscode


from fastai.callback.wandb import * #importar a bibliorteca do wheigts and bias

# 1. Inicializa o wandb (substitua pelos seus dados de projeto)
wandb.init(project="tiny-imagenet-squeezenet", name="treino-squeezenet") #coloca isso a mais no projeto

# 2. Treinamento com o WandbCallback integrado
print("Iniciando o treinamento...")
learn.fit_one_cycle(15, 1e-3, cbs=WandbCallback()) 

# 3. Finaliza a sessão do wandb
wandb.finish()

