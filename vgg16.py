
from pathlib import Path
import torchvision.models as tv_models
import torch.nn as nn
from fastai.vision.all import *
import wandb

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
dls = tiny_datablock.dataloaders(path / 'train', bs=128, num_workers=0)


print("DataLoaders criados com sucesso!")
print(f"Total de classes: {len(dls.vocab)}")
print(f"Imagens de Treino: {len(dls.train_ds)} | Validação: {len(dls.valid_ds)}")

# 5. VGG-16 com Batch Normalization e pesos pré-treinados
base_model = tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.DEFAULT)

# Ajusta a última camada linear do classificador da VGG para as 200 classes do Tiny ImageNet
num_classes = len(dls.vocab)
base_model.classifier[6] = nn.Linear(base_model.classifier[6].in_features, num_classes)
# 6. Criação do Learner
learn = Learner(
   dls,
   base_model,
   metrics=[accuracy, top_k_accuracy],
   loss_func=CrossEntropyLossFlat()
)

# 7. Treinamento com wheighst and bias
from fastai.callback.wandb import *
wandb.login(key="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")

wandb.init(project="tiny-imagenet-vgg16", name="treino-vgg16")
print("Iniciando o treinamento...")
learn.fit_one_cycle(15, 1e-3, cbs=WandbCallback())
wandb.finish()



# 8. Salvar o modelo treinado
learn.save('mobilenet_tinyimagenet_15ep')
print("Treinamento finalizado e modelo salvo!")

# Salva o learner completo em um arquivo portável (.pkl)
learn.export('modelo_vgg16_tinyimagenet.pkl')