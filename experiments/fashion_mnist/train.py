import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
from skimage.transform import resize
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

train_csv = pd.read_csv(sys.argv[1])
test_csv = pd.read_csv(sys.argv[2])
valid_csv = pd.read_csv(sys.argv[3])

class FashionDataset(Dataset):
    """Class to build a dataset from FashionMNIST using Pytorch class Dataset."""

    def __init__(self, data, transform=None):
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label = []
        image = []

        for i in self.fashion_MNIST:
            # first column is of labels.
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        m = np.mean(image)
        std = np.std(image)
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray((np.array(image)-m)/std).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        image = self.transform(resize(image, (224,224))/255)

        return image, label

    def __len__(self):
        return len(self.images)


class MnistResNet(nn.Module):
    """Class that edits Resnet50 to do FashionMNIST classification."""

    def __init__(self):
        super(MnistResNet, self).__init__()

        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model = models.resnet50(pretrained=True)

        # Change the input layer to take Grayscale image, instead of RGB images.
        # Hence in_channels is set as 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)


my_resnet = MnistResNet()


def calculate_metric(metric_fn, true_y, pred_y):
    """Calculate a metric on the outputs and ground truth
    :param metric_fn: metric function to calculate
    :param true_y: ground truth labels
    :param pred_y: predicted labels
    :return: the output of the applie dmetric_fn
    """
    if metric_fn == accuracy_score:
        return metric_fn(true_y, pred_y)
    else:
        return metric_fn(true_y, pred_y, average="macro")


def print_scores(p, r, f1, a, batch_size):
    """Print the P/R/F1 and Accuracy in a readable format
    :param p: precision
    :param r: recall
    :param f1: F1 between precision and recall
    :param a: accuracy
    :param batch_size: batch size used
    """
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")


# Transform data into Tensor that has a range from 0 to 1
train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))
valid_set = FashionDataset(valid_csv, transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=100, shuffle=True)


# model
model = MnistResNet().to(device)

# params
epochs = 5
batch_size = 100


# loss function and optimizer
loss_function = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

start_ts = time.time()

losses = []
batches = len(train_loader)
val_batches = len(valid_loader)
test_batches = len(test_loader)

# loop for every epoch (training + evaluation)
for epoch in range(epochs):
    total_loss = 0

    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    # ----------------- TRAINING  --------------------
    # set model to training
    model.train()

    for i, data in progress:
        X, y = data[0].to(device), data[1].to(device)

        # training step for single batch
        model.zero_grad()
        outputs = model(X)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        # updating progress bar
        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))

    # releasing unnecessary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------- VALIDATION  -----------------
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []

    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            X, y = data[0].to(device), data[1].to(device)

            outputs = model(X) # this gets the prediction from the network

            val_losses += loss_function(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction

            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )
        test_losses = 0
        precision2, recall2, f12, accuracy2 = [], [], [], []
        for i, data in enumerate(test_loader):
            X, y = data[0].to(device), data[1].to(device)

            outputs = model(X) # this get's the prediction from the network

            test_losses += loss_function(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction

            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision2, recall2, f12, accuracy2),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )

    print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}, test loss: {test_losses/test_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    print_scores(precision2, recall2, f12, accuracy2, test_batches)
    losses.append(total_loss/batches) # for plotting learning curve
print(f"Training time: {time.time()-start_ts}s")













