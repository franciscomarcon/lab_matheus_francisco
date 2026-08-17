from pathlib import Path
import torch
import torch.nn as nn
from fastai.vision.all import *

# -------------------------------------------------------------
# 1. Definição das Operações Binarizadas (1-bit com STE)
# -------------------------------------------------------------
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1.0] = 0.0
        return grad_input

def binarize(x):
    return BinarizeSTE.apply(x)

class BinarizedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(
            binarize(input), binarize(self.weight), self.bias, 
            self.stride, self.padding, self.dilation, self.groups
        )

class BinarizedLinear(nn.Linear):
    def forward(self, input):
        return nn.functional.linear(binarize(input), binarize(self.weight), self.bias)

# -------------------------------------------------------------
# 2. Arquitetura BNN para CIFAR-10 (10 Classes)
# -------------------------------------------------------------
class BNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Primeira camada em FP32 para preservar os canais RGB
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.Hardtanh()

        # Camadas Intermediárias Binarizadas (1-bit)
        self.bconv2 = BinarizedConv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.bconv3 = BinarizedConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.bconv4 = BinarizedConv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.blinear1 = BinarizedLinear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        
        # Última camada linear em FP32
        self.fc_out = nn.Linear(256, num_classes)
        self.act = nn.Hardtanh()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.bconv2(x)))
        x = self.pool1(x)

        x = self.act(self.bn3(self.bconv3(x)))
        x = self.act(self.bn4(self.bconv4(x)))
        x = self.pool2(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.bn_fc(self.blinear1(x)))
        return self.fc_out(x)

# -------------------------------------------------------------
# 3. Pipeline de Dados e Treinamento
# -------------------------------------------------------------
def main():
    print("Baixando/Carregando dataset CIFAR-10...")
    cifar_path = untar_data(URLs.CIFAR)

    dls = ImageDataLoaders.from_folder(
        cifar_path,
        train='train',
        valid='test',
        item_tfms=Resize(32),
        batch_tfms=aug_transforms(mult=1.0, do_flip=True),
        bs=64,
        num_workers=2
    )

    print(f"Dataset pronto! Classes: {len(dls.vocab)}")
    print(f"Treino: {len(dls.train_ds)} imagens | Validação: {len(dls.valid_ds)} imagens\n")

    # Instancia a rede
    model = BNN_CIFAR(num_classes=len(dls.vocab))

    # Cria o Learner com as métricas de tempo, loss e acurácia
    learn = Learner(
        dls,
        model,
        metrics=[accuracy, error_rate],
        loss_func=CrossEntropyLossFlat()
    )

    epochs = 15
    lr = 2e-3
    print(f"Iniciando treinamento da BNN no CIFAR-10 ({epochs} épocas)...")
    learn.fit_one_cycle(epochs, lr)

    learn.save("bnn_cifar10_15ep")
    print("\nTreinamento concluído e pesos salvos em 'models/bnn_cifar10_15ep.pth'!")

if __name__ == "__main__":
    main()