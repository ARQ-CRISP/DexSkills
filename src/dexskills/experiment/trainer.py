import lightining as L
import torch
from dexskills.data.dataset import Dataset
from dexskills.model.autoencoder import AutoEncoder
from dexskills.model.classifier import Classifier
from dexskills.model.composite import AutoEncoderClassifier
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder(
    (144 - 70 + 3 + 3 + 3) * 3 + (23 + 4) * 3, bottleneck_size=512
).to(device)

classifier = CustomNetwork(num_classes, input_size=512).to(device)

composite_model = AutoEncoderClassifier(autoencoder, classifier)

train_loader = DataLoader(Dataset(), batch_size=1, shuffle=True)

trainer = L.Trainer(accelerator="auto", devices="auto", strategy="auto")


if __name__ == "__main__":
    trainer.fit(composite_model, train_dataloaders=train_loader)
