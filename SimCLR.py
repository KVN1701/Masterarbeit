import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Subset
import torchvision
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform


class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        accuracy = Accuracy()
        acc = accuracy(z1, x1, task='multiclass')
        self.log('val_loss', loss)
        self.log('accuracy', acc, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        acccuracy = Accuracy(task='multiclass')
        acc = acccuracy(z1, x1)
        self.log('val_loss', loss)
        self.log('accuracy', acc, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        acccuracy = Accuracy(task='multiclass')
        acc = acccuracy(z1, x1)
        self.log('val_loss', loss)
        self.log('accuracy', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


def split_dataset(dataset, val_split=0.2):
    train_idx, val_indx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_indx)
    return datasets


torch.set_float32_matmul_precision('medium') # alternativ medium, da 4070ti tensor cores hat. Macht training schneller aber weniger genau

model = SimCLR()

transform = SimCLRTransform(input_size=32)

dataset = LightlyDataset('datasets/ubfc', transform=transform)
datasets = split_dataset(dataset)


dataloader_train = torch.utils.data.DataLoader(
    datasets['train'],
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=8,
    persistent_workers=True # beschleunigt den Trainingsprozess, bei erster epoche langes laden danach keine Wartezeit
)

dataloader_validate = torch.utils.data.DataLoader(
    datasets['val'],
    num_workers = 23,
    persistent_workers=True
)

if __name__ == '__main__':
    trainer = pl.Trainer(log_every_n_steps=2, max_epochs=10, devices=1, accelerator='gpu')
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_validate)
    trainer.test(model=model, dataloaders=dataloader_validate, ckpt_path='last')