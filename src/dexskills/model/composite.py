import lightning as pl
import torch
from dexskills.data.constants import label_names
from dexskills.model.autoencoder import AutoEncoder, Encoder
from dexskills.model.classifier import Classifier
from sklearn.preprocessing import OneHotEncoder
from tensorboardX import SummaryWriter


class AutoEncoderClassifier(pl.LightningModule):
    def __init__(
        self,
        autoencoder: AutoEncoder,
        classifier: Classifier,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.classifier = classifier

        self.save_hyperparameters("learning_rate")
        self.reconstruction_loss = torch.nn.MSELoss()
        self.label_loss = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter("DexSkill_model_training")
        self.label_encoder = OneHotEncoder(categories=[label_names], sparse=False)
        self.label_encoder.fit(label_names.reshape(-1, 1))

        self.automatic_optimization = False

    def forward(self, x):
        encoded = self.autoencoder.encoder(x)
        decoded = self.autoencoder.decoder(encoded)
        classified = self.classifier(encoded)
        return decoded, classified

    def training_step(self, batch, batch_idx):
        # x, y = batch
        en_optimizer, de_optimizer, cl_optimizer = self.optimizers()

        state_input = batch["state_input"].squeeze(0).float()
        state_output = batch["state_output"].squeeze(0).float()
        feature_input = batch["feature_input"].squeeze(0).float()
        feature_output = batch["feature_output"].squeeze(0).float()

        AE_input = torch.cat((feature_input, state_input), dim=1)
        AE_output = torch.cat((feature_output, state_output), dim=1)

        label = torch.tensor(
            self.label_encoder.transform(np.array(data["label"]).reshape(-1, 1))
        )

        decoded, classified = self(AE_input)

        en_optimizer.zero_grad()
        de_optimizer.zero_grad()
        cl_optimizer.zero_grad()

        reconstruction_loss = self.reconstruction_loss(decoded, AE_output)
        label_loss = self.label_loss(classified, label)
        (loss := reconstruction_loss + label_loss).backward()

        en_optimizer.step()
        de_optimizer.step()
        cl_optimizer.step()

        preds_class_indices = classified.argmax(dim=1).cpu().detach().numpy()

        self.log(f"{label_loss = :.2f} + {reconstruction_loss = : .2f} = {loss: .2f}")
        self.writer.add_scalar(
            "Reconstruction Loss/Train", reconstruction_loss, self.global_step
        )
        self.writer.add_scalar(
            "Classification Loss/Train", label_loss, self.global_step
        )
        self.writer.add_scalar("Loss/Train", loss, self.global_step)
        return loss

    def configure_optimizers(self):
        en_optimizer = torch.optim.Adam(
            params=self.encoder.parameters(), lr=self.hparams.learning_rate
        )
        de_optimizer = torch.optim.Adam(
            params=self.decoder.parameters(), lr=self.hparams.learning_rate
        )
        cl_optimizer = torch.optim.Adam(
            params=self.classifier.parameters(), lr=self.hparams.learning_rate
        )
        return [en_optimizer, de_optimizer, cl_optimizer]

    def on_train_end(self):
        # Close the writer when training ends
        self.writer.close()


class EncoderClassifier(pl.LightningModule):
    def __init__(self, encoder: Encoder, classifier: Classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        encoded = self.encoder(x)
        classified = self.classifier(encoded)
        return decoded, classified
