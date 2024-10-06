import numpy as np
import re
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Small epsilon value to prevent log(0) issues
EPSILON = 1e-10

def normalize_emotion_name(emotion):
    return "Neutral" if emotion.lower() in ["neutral state", "neutral"] else emotion

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def extract_relaxed_label_predictions(log_data):
    y_true = []
    y_pred = []
    skipped_count = 0

    for block in log_data.strip().split('----------------------------------------'):
        id_match = re.search(r'ID: (.+)', block)
        prediction_match = re.search(r'Prediction: (.+)', block)
        emotions_match = re.search(r'Emotions: (.+)', block)
        
        if id_match and prediction_match and emotions_match:
            entry_id = id_match.group(1).strip()

            prediction_str = prediction_match.group(1)
            emotions_str = emotions_match.group(1)

            # Normalize and get the ground truth emotions
            emotion_list = [normalize_emotion_name(label.strip()) for label in emotions_str.split(',')]
            emotion_counts = defaultdict(int)
            for emotion in emotion_list:
                emotion_counts[emotion] += 1

            # Calculate distribution of emotions
            emotion_distribution = {k: v / len(emotion_list) * 100 for k, v in emotion_counts.items()}

            # Ground truth is now all emotions in the emotion list
            ground_truth_set = set(emotion_list)

            # Check if the prediction is non-standard (e.g., contains no percentages or is in a strange format)
            if not any(char.isdigit() for char in prediction_str):
                skipped_count += 1
                continue  # Skip non-standard predictions

            # Parse the predictions and find the highest percentage predicted label
            prediction_dict = {}
            for pred in prediction_str.split(','):
                if ':' in pred:
                    emotion, value = pred.split(':', 1)
                    emotion = normalize_emotion_name(emotion.strip())
                    value = re.search(r'\d+', value.strip())
                    if value:
                        prediction_dict[emotion] = int(value.group())

            if prediction_dict:
                highest_predicted_label = max(prediction_dict, key=prediction_dict.get)

                # Check if the predicted label exists in the ground truth set
                if highest_predicted_label in ground_truth_set:
                    y_true.append(highest_predicted_label)
                    y_pred.append(highest_predicted_label)
                else:
                    # If no match, consider the ground truth as the most frequent emotion and compare
                    majority_emotion, _ = max(emotion_counts.items(), key=lambda item: item[1])
                    y_true.append(majority_emotion)
                    y_pred.append(highest_predicted_label)

                # Print the details
                print(f"ID: {entry_id}")
                print(f"  Emotions in Ground Truth: {emotions_str}")
                print(f"  Emotion Distribution: {emotion_distribution}")
                print(f"  Ground Truth Set: {ground_truth_set}")
                print(f"  Prediction Distribution: {prediction_dict}")
                print(f"  Predicted Label: {highest_predicted_label}\n")

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    labels = ["Anger", "Happiness", "Sadness", "Neutral"]
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate precision, recall, F1-score (unweighted and weighted)
    precision_unweighted, recall_unweighted, f1_unweighted, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # Print overall results
    print(f"Total relaxed-label predictions: {len(y_true)}")
    print(f"Skipped non-standard entries: {skipped_count}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nPrecision, Recall, F1 (Unweighted):")
    print(f"  Precision: {precision_unweighted:.4f}")
    print(f"  Recall: {recall_unweighted:.4f}")
    print(f"  F1-Score: {f1_unweighted:.4f}")
    print("\nPrecision, Recall, F1 (Weighted):")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall: {recall_weighted:.4f}")
    print(f"  F1-Score: {f1_weighted:.4f}")

    # Plot the confusion matrix
    plot_confusion_matrix(conf_matrix, labels)

    return accuracy, conf_matrix, precision_unweighted, recall_unweighted, f1_unweighted, precision_weighted, recall_weighted, f1_weighted

# Step 1: Read the log data from the file
with open('C:/Users/julev/Desktop/ltuAPI/LTU_Final_API/IEMOCAP/prediction_log_final.txt', 'r') as file:
    log_data = file.read()

# Step 2: Call the function with the log data
accuracy, conf_matrix, precision_unweighted, recall_unweighted, f1_unweighted, precision_weighted, recall_weighted, f1_weighted = extract_relaxed_label_predictions(log_data)
