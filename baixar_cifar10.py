from fastai.vision.all import untar_data, URLs
cifar_path = untar_data(URLs.CIFAR)
print("CIFAR-10 pronto em:", cifar_path)