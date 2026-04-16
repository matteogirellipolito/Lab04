import torch
import wandb
from dataset.cifar10 import train_loader, test_loader, input_size, n_channels, output_size
from models.models import FC2Layer, CNN
from utils.utils import fit, get_model_optimizer, count_parameters

wandb.init(project="Lab04")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

epochs = 4
n_hidden = 9
n_features = 6

models_accuracy = {}  #  dizionario per salvare le accuracy

# FNN
model_fnn = FC2Layer(input_size, n_channels, n_hidden, output_size)
model_fnn.to(device)
optimizer = get_model_optimizer(model_fnn)
models_accuracy['fnn'] = fit(epochs=epochs, train_dl=train_loader, test_dl=test_loader,
    model=model_fnn, opt=optimizer, tag='fnn', device=device)

# CNN
model_cnn = CNN(input_size, n_channels, n_features, output_size)
model_cnn.to(device)
optimizer = get_model_optimizer(model_cnn)
models_accuracy['cnn'] = fit(epochs=epochs, train_dl=train_loader, test_dl=test_loader,
    model=model_cnn, opt=optimizer, tag='cnn', device=device)

# Scrambled FNN
perm = torch.randperm(input_size)
model_scrambled_fnn = FC2Layer(input_size, n_channels, n_hidden, output_size)
model_scrambled_fnn.to(device)
optimizer = get_model_optimizer(model_scrambled_fnn)
models_accuracy['scrambled_fnn'] = fit(epochs=epochs, train_dl=train_loader, test_dl=test_loader,
    model=model_scrambled_fnn, opt=optimizer, tag='scrambled_fnn', perm=perm, device=device)

# Scrambled CNN
model_scrambled_cnn = CNN(input_size, n_channels, n_features, output_size)
model_scrambled_cnn.to(device)
optimizer = get_model_optimizer(model_scrambled_cnn)
models_accuracy['scrambled_cnn'] = fit(epochs=epochs, train_dl=train_loader, test_dl=test_loader,
    model=model_scrambled_cnn, opt=optimizer, tag='scrambled_cnn', perm=perm, device=device)

# Salva il dizionario su file per eval.py
import json
with open("models_accuracy.json", "w") as f:
    json.dump(models_accuracy, f)