from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_metrics(model, test_dataloader, inception=False):
    true_labels = []
    predicted_labels = []
    accuracy_values = []
    correct = 0
    total = 0
    model.eval()
    tp = [0] * 5
    tn = [0] * 5
    fp = [0] * 5
    fn = [0] * 5

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if inception:
                _, predicted = torch.max(outputs, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # Calculate TP, TN, FP, FN
            for i in range(5):
                tp[i] += ((predicted == i) & (labels == i)).sum().item()
                tn[i] += ((predicted != i) & (labels != i)).sum().item()
                fp[i] += ((predicted == i) & (labels != i)).sum().item()
                fn[i] += ((predicted != i) & (labels == i)).sum().item()
    

    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy, precision, recall, f1, tp, tn, fp, fn, confusion_matrix(true_labels, predicted_labels)


def plot_metric(lst, title, line='', model="googlenet", save=False, show=True, output="", num_epochs=10, test_ratio=0.3):
    # Plot the accuracy values
    x_values = [i + 1 for i in range(len(lst)) if i % 10 == 0]
    lst = [lst[i] for i in range(len(lst)) if i % 10 == 0]
    plt.rcdefaults()
    plt.clf()
    plt.scatter(x_values, lst, marker='o')
    plt.ylim(0, 1)
    # Set the x-axis label and title
    plt.xlabel('Timestamp')
    plt.ylabel(title)
    plt.title(f"{model} test ratio={test_ratio} epochs={num_epochs}")
    # Define the desired tick locations and labels
    # xtick_locations = np.arange(0, len(lst), 50)
    # xtick_labels = [str(x) for x in xtick_locations]

    # Set the x-axis tick locations and labels
    # plt.xticks(xtick_locations, xtick_labels)
    if show:
        plt.show()
    if save:
        save_dir = os.path.join(output, model)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir,f'{title}'))

def plot_confusion_matrix(confusion_matrix_list, time, save=False, show=True, output=""):
    class_labels = ["BMP4", "CHIR", "DS", "DS+CHIR", "WT"]
    num_classes = len(class_labels)
    confusion_matrix = confusion_matrix_list[time-1]

    
    # Plot confusion matrix
    plt.clf()
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)  # Adjust font scale if necessary
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for timestamp {time}')
    
    if show:
        plt.show()
    
    if save:
        plt.savefig(os.path.join(output, f'confusion_{time}'))
# def plot_confusion_matrix(tps, tns, fps, fns, time, save=False, show=True, output=""):
#     TP, TN, FP, FN = tps[time-1], tns[time-1], fps[time-1], fns[time-1]
#     conf_matrix = np.array([[TN[0], FP[0], 0, 0, FN[0]],
#                         [FP[1], TN[1], 0, 0, FN[1]],
#                         [0, 0, TN[2], FP[2], FN[2]],
#                         [0, 0, FP[3], TN[3], FN[3]],
#                         [FP[4], 0, 0, 0, TN[4]]])
#     class_labels = ["BMP4", "CHIR", "DS", "DS+CHIR",  "WT"]
#     # Plot confusion matrix
#     plt.clf()
#     plt.figure(figsize=(10, 8))
#     sns.set(font_scale=1.2)  # Adjust font scale if necessary
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title(f'Confusion Matrix for timestamp {time}')
#     if show:
#         plt.show()
#     if save:
#         plt.savefig(os.path.join(output,f'confusion_{time}'))
