from torchvision.datasets import MNIST

train_data = MNIST('./data', True, download=True)
print(train_data[0][0])
