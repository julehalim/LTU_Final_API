import json
from collections import defaultdict, Counter
import re
import numpy as np
from dictances import kullback_leibler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Small epsilon value to prevent division by zero issues
EPSILON = 1e-10

# Bhattacharyya Coefficient function
def bhattacharyya_coefficient(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return np.sum(np.sqrt(p * q))

# R-squared calculation function
def R(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    epsilon = EPSILON
    covariance = np.cov(a, b, ddof=0)[0][1]
    var_a = np.var(a, ddof=0)
    var_b = np.var(b, ddof=0)
    return round((covariance ** 2) / (var_a * var_b + epsilon), 4)

# Helper function to normalize emotion names
def normalize_emotion_name(emotion):
    return "Neutral" if emotion.lower() in ["neutral state", "neutral"] else emotion

# Normalize function to calculate percentages from logits
def normalize_over_100(logits):
    total_sum = np.sum(logits) + EPSILON  # Ensure no division by zero
    normalized_logits = np.clip(logits, 0, None)  # Clip to ensure non-negative values
    total_normalized_sum = np.sum(normalized_logits) + EPSILON  # Recalculate total for non-negative values
    return [(logit / total_normalized_sum) * 100 for logit in normalized_logits]

# Function to extract the core part of the audio ID from the path
def extract_audio_id_from_path(audio_path):
    match = re.search(r'(Ses\d+[FM]_(impro|script)\d+[a-zA-Z]?(_\d+)?_[FM]\d+)', audio_path)
    if match:
        return match.group(0)
    match = re.search(r'(Ses\d+[FM]_(impro|script)\d+[a-zA-Z]?\d*)', audio_path)
    if match:
        session_part = match.group(0)
        file_id_match = re.search(r'_[FM]\d+', audio_path)
        if file_id_match:
            return session_part + file_id_match.group(0)
    return None

# Aggregate logits across targets for each audio ID
def aggregate_logits_by_id(log_data):
    aggregated_logits_by_id = defaultdict(lambda: defaultdict(list))
    for entry in log_data:
        audio_id = extract_audio_id_from_path(entry["audio_id"])  # Extract the audio ID
        logits_for_targets = entry.get("logits_for_targets", {})
        for emotion, logits in logits_for_targets.items():
            emotion_normalized = normalize_emotion_name(emotion)
            aggregated_logits_by_id[audio_id][emotion_normalized].extend(logits)
    return aggregated_logits_by_id

# Calculate the normalized distribution for each audio ID
def calculate_distribution(aggregated_logits_by_id):
    distribution_by_id = {}
    for audio_id, logits_dict in aggregated_logits_by_id.items():
        logits_sum = {}
        for emotion, logits in logits_dict.items():
            logits_sum[emotion] = np.sum(logits)
        emotions = list(logits_sum.keys())
        logits_values = list(logits_sum.values())
        normalized_probabilities = normalize_over_100(logits_values)  # Normalize to sum to 100
        distribution_by_id[audio_id] = dict(zip(emotions, normalized_probabilities))
    return distribution_by_id

# Function to calculate the ground truth distribution of emotions
def calculate_ground_truth_distribution(ground_truth_emotions):
    total = len(ground_truth_emotions)
    emotion_counts = Counter(ground_truth_emotions)
    ground_truth_distribution = {normalize_emotion_name(emotion): (count / total) * 100 for emotion, count in emotion_counts.items()}
    return ground_truth_distribution

# Function to calculate majority label from the ground truth
def get_majority_label(ground_truth_emotions):
    emotion_counts = Counter(ground_truth_emotions)
    majority_label = emotion_counts.most_common(1)[0][0]  # Get the most frequent emotion
    return normalize_emotion_name(majority_label)

# Plot confusion matrix
def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Load the log file with predicted logits
log_file = "LTU_Final_API/IEMOCAP/Logs/single_label_2024-09-28-15-02-50.json"  # Replace with your file path
with open(log_file, 'r') as file:
    log_data = json.load(file)

# Aggregate logits by audio ID and target emotions
aggregated_logits_by_id = aggregate_logits_by_id(log_data)

# Calculate the normalized distribution for each audio ID
distribution_by_id = calculate_distribution(aggregated_logits_by_id)

# Load the ground truth file
ground_truth_file = "LTU_Final_API/IEMOCAP/iemocap_ambiguous.json"  # Replace with your file path
with open(ground_truth_file, 'r') as file:
    ground_truth_data = json.load(file)

# Filter out the entries that need predictions
entries_needing_predictions = [entry for entry in ground_truth_data if entry["need_prediction"].lower() == "yes"]

# Initialize lists to store true and predicted labels
y_true = []
y_pred = []

# Iterate through entries needing predictions and compare ground truth and predicted distributions
for entry in entries_needing_predictions:
    audio_id = entry["id"]  # Ground truth ID (already in core format like 'Ses03M_impro08b_M000')
    ground_truth_emotions = entry["emotion"]

    # Calculate the ground truth distribution
    ground_truth_distribution = calculate_ground_truth_distribution(ground_truth_emotions)

    # Get the majority label from the ground truth
    majority_label = get_majority_label(ground_truth_emotions)

    predicted_distribution = distribution_by_id.get(audio_id, {})

    if predicted_distribution:
        # Find the predicted emotion with the highest percentage
        highest_pred_emotion = max(predicted_distribution, key=predicted_distribution.get)

        # Store the true and predicted labels
        y_true.append(majority_label)
        y_pred.append(highest_pred_emotion)

        # Print results
        print(f"Audio ID: {audio_id}")
        print(f"Ground Truth Majority Emotion: {majority_label}")
        print(f"Predicted Emotion: {highest_pred_emotion} with {predicted_distribution[highest_pred_emotion]:.2f}%")
        print()

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate confusion matrix
labels = ["Anger", "Happiness", "Sadness", "Neutral"]
conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

# Calculate precision, recall, F1-score (unweighted and weighted)
precision_unweighted, recall_unweighted, f1_unweighted, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

# Print metrics
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

# Plot confusion matrix
plot_confusion_matrix(conf_matrix, labels)
