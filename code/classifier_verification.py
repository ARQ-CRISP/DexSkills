
import gc
import torch
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
# from segmentation import CustomNetwork
from network import  Autoencoder,  CustomNetwork
# from data_preparation_segmentation_time_series import RobotDataset
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import json
import os

import re

def extract_number(f):
    s = re.findall("\d+", f)
    return (int(s[0]) if s else -1, f)

class MyDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = sorted(os.listdir(data_dir), key=extract_number)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        # Load the preprocessed data
        data = torch.load(file_path)
        return data


segmented_ground_truth_total = []
segmented_prediction_total = []
segmented_latent_features_total = []
#######save parameter
def save_task_metrics(task_name, task_metrics, filename='0.1_dataset_eval.json'):
    # Try to load existing metrics from the file
    try:
        with open(filename, 'r') as file:
            all_metrics = json.load(file)
    except FileNotFoundError:
        all_metrics = {}

    # Update the metrics with the new task's results
    all_metrics[task_name] = task_metrics

    # Save the updated metrics back to the file
    with open(filename, 'w') as file:
        json.dump(all_metrics, file, indent=4)


def calculate_evaluation_metrics(segmented_ground_truth, segmented_predictions):
    """
    Calculate and return evaluation metrics for the given ground truth and predictions.
    """
    accuracy = accuracy_score(segmented_ground_truth, segmented_predictions)
    precision = precision_score(segmented_ground_truth, segmented_predictions, average='macro', zero_division=0)
    recall = recall_score(segmented_ground_truth, segmented_predictions, average='macro', zero_division=0)
    f1 = f1_score(segmented_ground_truth, segmented_predictions, average='macro')
    # Assuming calculate_iou is correctly defined to handle your data
    iou_scores = calculate_iou(segmented_ground_truth, segmented_predictions, np.unique(segmented_ground_truth))
    average_iou = np.mean(list(iou_scores.values()))

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Average IoU': average_iou
    }




# Load the JSON structured labels
with open('./json_file/LH_final.json', 'r') as file:
    structured_labels = json.load(file)


def prediction_eval(dataset_identifier, structured_labels, preds_class_indices, sample_rate=20, latent_feature=None):
    # Calculate the total duration to determine the size of the NumPy array
    # Assuming the last interval's end_time represents the total duration
    if not structured_labels.get(dataset_identifier):
        return np.array([])  # Return an empty array if the dataset isn't found
    total_duration = structured_labels[dataset_identifier][-1]["end_time"]
    inistal_time = structured_labels[dataset_identifier][0]["start_time"]
    num_samples = int(total_duration * sample_rate)
    # Create an empty array of strings, with a default label if necessary
    # label_array = np.full(num_samples, "none", dtype=object)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # Fill the array with labels for each defined interval
    i = 0
    segmented_prediction = []
    segmented_ground_truth = []
    segmented_latent_features = []

    for interval in structured_labels[dataset_identifier]:
        start_index = int(interval["start_time"] * sample_rate)  
        end_index = int(interval["end_time"] * sample_rate)  
        # label_array[start_index:end_index] = interval["label"]
        label_index = encoder_new.transform([[interval["label"]]]).argmax()

        ##########

        segmented_prediction.append(preds_class_indices[start_index:end_index])
        segmented_ground_truth.append(np.full(end_index-start_index, label_index))

        color = colors[i % len(colors)]
        i+=1


        ######TSNE#####
        if latent_feature is not None:
            segmented_latent_features.append(latent_feature[start_index:end_index].cpu().numpy())

    segmented_prediction = np.concatenate(segmented_prediction)
    segmented_ground_truth = np.concatenate(segmented_ground_truth)

    segmented_ground_truth_total.append(segmented_ground_truth)
    segmented_prediction_total.append(segmented_prediction)

    if latent_feature is not None:
        segmented_latent_features = np.concatenate(segmented_latent_features)
        segmented_latent_features_total.append(segmented_latent_features)


    accuracy = accuracy_score(segmented_ground_truth, segmented_prediction)
    precision = precision_score(segmented_ground_truth, segmented_prediction, average='macro', zero_division=0)
    recall = recall_score(segmented_ground_truth, segmented_prediction, average='macro', zero_division=0)
    f1 = f1_score(segmented_ground_truth, segmented_prediction, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


    classes = np.unique(segmented_ground_truth)  # Adjust according to your dataset
    iou_scores = calculate_iou(np.array(segmented_ground_truth), np.array(segmented_prediction), classes)
    average_iou = np.mean(list(iou_scores.values()))

    print(f"Average IoU: {average_iou:.4f}")
    task_metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Average IoU': average_iou
    }
    # save_task_metrics(dataset_identifier, task_metrics)



# IoU needs a custom implementation based on your task's specifics



# Custom IoU calculation for classification 
def calculate_iou(y_true, y_pred, classes):
    iou_scores = {}
    for cls in classes:
        intersection = np.logical_and(y_true == cls, y_pred == cls).sum()
        union = np.logical_or(y_true == cls, y_pred == cls).sum()
        iou_scores[cls] = intersection / union if union != 0 else 0
    return iou_scores


torch.manual_seed(3407)
# The label name and sequence defined when createing dataset is different with the final version in the paper. Here we use a transformation to convert the label name to the final version.
label_names_new = np.array(['Reach', 'Go_back_to_setup', 'Pre_touch', 'Touch_object','Flip' , 'Push', 'Pull', 'Pre_grasp', 'Grasp' , 'Lift_with_grasp', 'Transport_forward', 'Transport_down',  'Pre_rotate', 'Rotate', 'Shake_a', 'Shake_b', 'Twist', 'Side_place', 'Pour',  'Release'])
label_names_old = np.array(['Reach', 'Flip', 'Touch_object', 'Push','Pull', 'Pre_grasp', 'Grasp' , 'Lift_with_grasp', 'Transport_forward', 'Transport_down', 'Release', 'Go_back_to_setup', 'Pre_rotate', 'Rotate', 'Shake_a', 'Shake_b', 'Twist', 'Pre_touch', 'Pour', 'Side_place' ])
# Reshaping to 2D array as required by OneHotEncoder
labels_old = label_names_old.reshape(-1, 1)
labels_new = label_names_new.reshape(-1, 1)
# Initialize OneHotEncoder with specified categories in the correct order
encoder_old = OneHotEncoder(categories=[label_names_old], sparse=False)
encoder_new = OneHotEncoder(categories=[label_names_new], sparse=False)
# Fit and transform
one_hot_labels_old = encoder_old.fit_transform(labels_old)
one_hot_labels_new = encoder_new.fit_transform(labels_new)
num_classes = len(label_names_new)

def exponential_moving_average(predictions, alpha=0.3):
    """
    Compute the exponential moving average of a series of numbers.

    :param predictions: List or numpy array of predictions (numerical).
    :param alpha: Smoothing factor, between 0 and 1. Higher alpha discounts older observations faster.
    :return: Numpy array of the EMA.
    """
    ema = [predictions[0]]  # Start the EMA with the first prediction
    for p in predictions[1:]:
        ema.append(alpha * p + (1 - alpha) * ema[-1])
    return np.array(ema)
#  
AE_NET = Autoencoder( input_size = (144-70+3+3+3)*3+(23+4)*3, bottleneck_size=512).cuda()
AE_NET.load_state_dict(torch.load("./trained_policy/AE_20skills_0.1s.pth"))

AE_NET.eval()
Seg_NET = CustomNetwork(num_classes,  input_size=512).cuda()
Seg_NET.load_state_dict(torch.load("./trained_policy/Seg_20skills_0.1s.pth"))
Seg_NET.eval()

criterion = torch.nn.MSELoss()

#######TSNE#######

dataset = MyDataset('./LH_dataset')
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

for idx, data_value in enumerate(train_loader):
    dataset_name = f"LH_{idx+1}"
    with torch.no_grad():
        n_title = np.array(['Reach', 'Setup Position', 'PreTouch', 'Touch','Flip', 'Wipe Forth', 'Wipe Back' , 'PreGrasp', 'Grasp', 'Lift with Grasp', 'Transport Forward', 'Place', 'PreRotate', 'Rotate', 'Shake Up', 'Shake Down', 'Twist', 'Vertical Place', 'Pour', 'Release' ])
        state_input = data_value['state_input'].squeeze(0).float().cuda() 
        state_output = data_value['state_output'].squeeze(0).float().cuda()
        feature_input = data_value['feature_input'].squeeze(0).float().cuda()
        feature_output = data_value['feature_output'].squeeze(0).float().cuda()
        AE_input = torch.cat((feature_input, state_input), dim=1)


        AE_output = torch.cat((feature_output, state_output), dim=1)

        recover_feature= AE_NET(AE_input)
        loss_reconstruction = criterion(recover_feature.squeeze(1), AE_output)
        print("loss_reconstruction: {}".format(loss_reconstruction))
        latent_feature = AE_NET.get_latent_vector(AE_input)
        classifier_out = Seg_NET(latent_feature)
        label = encoder_old.transform(np.array(data_value['label']).reshape(-1, 1))
        label = torch.tensor(label).cuda()
        ###### Median filter
        probabilities = torch.nn.functional.softmax(classifier_out, dim=1).cpu().numpy()
        window_size = 30
        max_probs = torch.zeros(probabilities.shape[0] - window_size + 1, 1)
        for j in range(classifier_out.shape[0] - window_size):
            sum_prob = np.sum(probabilities[j:j+window_size], axis=0)
            max_probs[j] = np.argmax(sum_prob)  # Store the max values
        old_name = label_names_old[max_probs.int()]
        new_name = encoder_new.fit_transform(old_name)
        new_index = new_name.argmax(axis=1)
        labels = torch.argmax(label, dim=1)
        class_indices = labels.cpu().detach().numpy()
        true_class_indices = prediction_eval(dataset_name, structured_labels, new_index.reshape(-1),  20)

    print(f"Finished processing {dataset_name}")

segmented_prediction_total = np.concatenate(segmented_prediction_total)
segmented_ground_truth_total = np.concatenate(segmented_ground_truth_total)


metrics = calculate_evaluation_metrics(segmented_ground_truth_total, segmented_prediction_total)
print(f"Metrics for {dataset_name}: {metrics}")
save_task_metrics("Total_results", metrics)


###Confusion matrix###
class_labels = np.unique(segmented_ground_truth_total)
conf_matrix = confusion_matrix(segmented_ground_truth_total, segmented_prediction_total)
df_conf_matrix = pd.DataFrame(conf_matrix, index=n_title, columns=n_title)



