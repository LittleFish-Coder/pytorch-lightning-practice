# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
        

# pl
class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.my_accuracy = Accuracy()
        # self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.my_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy, "train_f1_score": f1_score}, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "accuracy": accuracy, "f1_score": f1_score}
    
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        # loss = F.cross_entropy(scores, y)
        loss = self.loss_fn(scores, y)
        return loss, scores, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
        


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 1 * 28 * 28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# load data
entire_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
# train validation split
train_dataset, val_dataset = random_split(entire_dataset, [50000, 10000])
# loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# initialize
model = NN(input_size=input_size, num_classes=num_classes)
model = model.to(device)
print(f"The model is on device: {model.device}")

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# trainer
trainer = pl.Trainer(accelerator='gpu', devices=1, min_epochs=1, max_epochs=num_epochs, precision=16)
trainer.fit(model, train_loader, val_loader)
trainer.validate(model, val_loader)
trainer.test(model, test_loader)

# After trainer, the model is on device: cpu
print(f"The model is on device: {model.device}")

# check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)

            # flatten
            data = data.reshape(data.shape[0], -1)

            scores = model(data)

            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        acc = num_correct / num_samples
        

    model.train()
    return acc


print(f"Accuracy on training set: {check_accuracy(train_loader, model):.4f}")
print(f"Accuracy on validation set: {check_accuracy(val_loader, model):.4f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model):.4f}")