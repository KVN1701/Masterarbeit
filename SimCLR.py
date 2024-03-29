import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Subset
import torchvision
from sklearn.model_selection import train_test_split
from torch.nn.functional import cosine_similarity


from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

training_losses = []


class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 1024, 1024) # ? input=Bildgröße, andere: 512, typische Ausgabe-/ hidden-layer-Dimension -> ruft Fehler hervor, warum?
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
        acc = cosine_similarity(z0, z1)
        self.log("Train loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('accuracy', acc.mean(), on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        acc = cosine_similarity(z0, z1)
        self.log("Train loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('accuracy', acc.mean(), on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss
        
    def test_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        acc = cosine_similarity(z0, z1)
        self.log('test_loss', loss, batch_size=batch_size)
        self.log('accuracy', acc.mean(), on_epoch=True, batch_size=batch_size)
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


torch.set_float32_matmul_precision('high') # alternativ medium, da 4070ti tensor cores hat. Macht training schneller aber weniger genau

model = SimCLR()

transform = SimCLRTransform(input_size=256)

dataset = LightlyDataset('datasets/ubfc', transform=transform)
datasets = split_dataset(dataset)
batch_size = 32


dataloader_train = torch.utils.data.DataLoader(
    datasets['train'],
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    persistent_workers=True # beschleunigt den Trainingsprozess, bei erster epoche langes laden danach keine Wartezeit
)

dataloader_validate = torch.utils.data.DataLoader(
    datasets['val'],
    batch_size=batch_size,
    num_workers = 23,
    persistent_workers=True
)

if __name__ == '__main__':
    trainer = pl.Trainer(log_every_n_steps=50, max_epochs=200, devices=1, accelerator='gpu')
    trainer.fit(model=model, train_dataloaders=dataloader_train)
    trainer.test(model=model, dataloaders=dataloader_validate, ckpt_path='best')