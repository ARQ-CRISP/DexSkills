import logging

import lightning as L
import torch
from dexskills.data.constants import label_names
from dexskills.data.dataset import DexSkillsDataset, LongHorizonTasksDataset
from dexskills.model.autoencoder import AutoEncoder
from dexskills.model.classifier import ClassifierNet
from dexskills.model.composite import AutoEncoderClassifier
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

torch.manual_seed(3407)

INPUT_FEATURE_DIM = (144 - 70 + 3 + 3 + 3) * 3 + (23 + 4) * 3
BOTTLENECK_DIM = 512
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = AutoEncoder(INPUT_FEATURE_DIM, bottleneck_size=BOTTLENECK_DIM).to(device)
classifier = ClassifierNet(num_classes=len(label_names), input_size=BOTTLENECK_DIM).to(
    device
)

composite_model = AutoEncoderClassifier(autoencoder, classifier)

train_loader = DataLoader(DexSkillsDataset(), batch_size=1, shuffle=True, num_workers=4)
test_loader = DataLoader(
    LongHorizonTasksDataset(), batch_size=1, shuffle=False, num_workers=4
)


checkpoint_callback = ModelCheckpoint(
    monitor="total_loss",  # Name of the metric to monitor
    dirpath="checkpoints/",  # Directory where checkpoints will be saved
    filename="skill-segmentation-net-{epoch:02d}-{total_loss:.2f}",
    save_top_k=3,  # Save the top 3 models according to val_loss
    mode="min",  # Minimize val_loss
    every_n_epochs=1,  # Checkpoint frequency
)


trainer = L.Trainer(
    accelerator="auto",
    devices="auto",
    strategy="auto",
    max_epochs=1000,
    callbacks=[checkpoint_callback],
)


if __name__ == "__main__":
    trainer.fit(composite_model, train_dataloaders=train_loader)
