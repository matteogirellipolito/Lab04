import torch
from torchvision import datasets, transforms

image_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data", train=True, download=True, transform=image_transforms),
    batch_size=64,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data", train=False, transform=image_transforms),
    batch_size=1000,
    shuffle=True,
)

x, yy = next(iter(train_loader))
n_channels = x.shape[1]
input_size_w = x.shape[2]
input_size_h = x.shape[3]
input_size = input_size_w * input_size_h
output_size = yy.max().item() + 1

output_classes = ('plane', 'car', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck')