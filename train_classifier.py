import torch
from torch.utils import data
from network import  Autoencoder, CustomNetwork
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from torch.utils import data
import os
from sklearn.metrics import precision_score

torch.manual_seed(3407)

class MyDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        # Load the preprocessed data
        data = torch.load(file_path)
        return data

label_names = np.array(['Reach', 'Flip', 'Touch_object', 'Push','Pull', 'Pre_grasp', 'Grasp' , 'Lift_with_grasp', 'Transport_forward', 'Transport_down', 'Release', 'Go_back_to_setup', 'Pre_rotate', 'Rotate', 'Shake_a', 'Shake_b', 'Twist', 'Pre_touch', 'Pour', 'Side_place' ])
labels = label_names.reshape(-1, 1)

# Initialize OneHotEncoder with specified categories in the correct order
label_encoder = OneHotEncoder(categories=[label_names], sparse=False)
# Fit and transform
one_hot_labels = label_encoder.fit_transform(labels)
num_classes = len(label_names)
epoch = 1000

# Define the network
AE_NET = Autoencoder(  (144-70+3+3+3 )*3 +(23+4)*3 , bottleneck_size=512).cuda()
AE_NET.train()
encoder_optimizer = torch.optim.Adam(AE_NET.encoder.parameters(), lr=0.001)
decoder_optimizer = torch.optim.Adam(AE_NET.decoder.parameters(), lr=0.001)
Seg_NET = CustomNetwork(num_classes, key_frame_output_size=8, input_size=512).cuda()
Seg_NET.train()
seg_net_optimizer = torch.optim.Adam(Seg_NET.parameters(), lr=0.001)

# Define the loss function
writer = SummaryWriter()
criterion = torch.nn.MSELoss()
label_loss = torch.nn.CrossEntropyLoss()

# Train the network
for i in range(epoch):
    dataset = MyDataset('./dataset')
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)


    pbar = tqdm(train_loader, total=len(train_loader))  # Initialize tqdm with the total number of batches
    precision = 0
    total_prediction_loss = 0
    for idx, data in enumerate(pbar):

        state_input = data['state_input'].squeeze(0).float().cuda() 
        state_output = data['state_output'].squeeze(0).float().cuda()
        feature_input = data['feature_input'].squeeze(0).float().cuda()
        feature_output = data['feature_output'].squeeze(0).float().cuda()
        AE_input = torch.cat((feature_input, state_input), dim=1)
        AE_output = torch.cat((feature_output, state_output), dim=1)


        encoder_optimizer.zero_grad()  # Clear gradients of shared encoder
        decoder_optimizer.zero_grad()   # Clear gradients of AE_NET decoder
        seg_net_optimizer.zero_grad()  # Clear gradients of SEG_NET specific layers

        recover_feature= AE_NET(AE_input)
        loss_reconstruction = criterion(recover_feature.squeeze(1), AE_output)

        latent_feature = AE_NET.get_latent_vector(AE_input)
        classifier_out = Seg_NET(latent_feature)
        label = label_encoder.transform(np.array(data['label']).reshape(-1, 1))
        label = torch.tensor(label).cuda()
        loss_classifier = label_loss(classifier_out, label)
        loss =  loss_classifier + loss_reconstruction
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        seg_net_optimizer.step()

        pbar.set_description(f"[Epoch {i+1}] Training {idx}/{len(train_loader)} Loss1: {loss_classifier.item():.4f}, Loss2: {loss_reconstruction.item():.4f}")
        writer.add_scalar("Loss/train", loss, global_step=i*len(train_loader)+idx)
        preds_class_indices = classifier_out.argmax(dim=1).cpu().detach().numpy()


        labels = torch.argmax(label, dim=1)

        true_class_indices = labels.cpu().detach().numpy()
        precision += precision_score(true_class_indices, preds_class_indices, average='macro', zero_division=0)
        total_prediction_loss += loss_classifier.item()
    if i> 990:
        torch.save(AE_NET.state_dict(), "./trained_policy_MLP/AE_256_0or1_no_temp_epoch_{}.pth".format(i))
        torch.save(Seg_NET.state_dict(), ".trained_policy_MLP/Seg_256_0or1_no_temp_epoch_{}.pth".format(i))
    print("total_prediction_loss: {}".format(total_prediction_loss))
    print("precision: {}".format(precision/7))

