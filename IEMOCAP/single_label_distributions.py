import json
from collections import defaultdict, Counter
import re
import numpy as np
from dictances import kullback_leibler

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

# Function to calculate Bhattacharyya, KL, and R-squared
def calculate_metrics(ground_truth_distribution, predicted_distribution):
    all_emotions = set(ground_truth_distribution.keys()).union(predicted_distribution.keys())
    
    actual_values = np.array([ground_truth_distribution.get(emotion, 0) + EPSILON for emotion in all_emotions])
    predicted_values = np.array([predicted_distribution.get(emotion, 0) + EPSILON for emotion in all_emotions])

    # Normalize both to sum to 1
    actual_values /= actual_values.sum()
    predicted_values /= predicted_values.sum()

    # Bhattacharyya Coefficient
    bhattacharyya = bhattacharyya_coefficient(predicted_values, actual_values)
    
    # Kullback-Leibler Divergence
    kl_divergence = kullback_leibler(
        {emotion: actual_values[i] for i, emotion in enumerate(all_emotions)},
        {emotion: predicted_values[i] for i, emotion in enumerate(all_emotions)}
    )

    # R-squared
    r_squared = R(actual_values, predicted_values)

    return bhattacharyya, kl_divergence, r_squared

# Load the log file with predicted logits
log_file = "LTU_Final_API\IEMOCAP\Logs\single_label_2024-09-28-15-02-50.json"  # Replace with your file path
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

# Initialize totals for metrics
total_bc = 0
total_kl = 0
total_r2 = 0
count = 0

# Iterate through entries needing predictions and compare ground truth and predicted distributions
for entry in entries_needing_predictions:
    audio_id = entry["id"]  # Ground truth ID (already in core format like 'Ses03M_impro08b_M000')
    ground_truth_emotions = entry["emotion"]

    # Calculate the ground truth distribution
    ground_truth_distribution = calculate_ground_truth_distribution(ground_truth_emotions)

    # Get the predicted distribution
    predicted_distribution = distribution_by_id.get(audio_id, {})

    # Merge the ground truth and predicted distributions, assigning 0% to missing emotions
    all_emotions = set(ground_truth_distribution.keys()).union(predicted_distribution.keys())
    merged_distribution = {}
    
    for emotion in all_emotions:
        gt_prob = ground_truth_distribution.get(emotion, 0)  # Ground truth, or 0% if missing
        pred_prob = predicted_distribution.get(emotion, 0)    # Predicted, or 0% if missing
        merged_distribution[emotion] = (gt_prob, pred_prob)

    # Print the ground truth and predicted probabilities
    print(f"Audio ID: {audio_id}")
    print("Ground Truth Distribution and Predicted Distribution:")
    for emotion, (gt_prob, pred_prob) in merged_distribution.items():
        print(f"  {emotion}: Ground Truth: {gt_prob:.2f}%, Predicted: {pred_prob:.2f}%")
    
    # If there's a predicted distribution available, calculate metrics
    if predicted_distribution:
        bhattacharyya, kl_divergence, r_squared = calculate_metrics(ground_truth_distribution, predicted_distribution)

        total_bc += bhattacharyya
        total_kl += kl_divergence
        total_r2 += r_squared
        count += 1

# Final average values for metrics
if count > 0:
    final_bc = total_bc / count
    final_kl = total_kl / count
    final_r2 = total_r2 / count

    print(f"\n  Bhattacharyya Coefficient: {final_bc:.4f}")
    print(f"  Kullback-Leibler Divergence: {final_kl:.4f}")
    print(f"  R-squared: {final_r2:.4f}\n")
else:
    print("No predictions were matched with ground truth entries.")
