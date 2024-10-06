import re
import sys
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract audio ID from path
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

# Ensure UTF-8 encoding is used
sys.stdout.reconfigure(encoding='utf-8')

# Function to normalize probabilities (so they sum to 1)
def normalize(probabilities):
    total = sum(probabilities)
    return [p / total for p in probabilities] if total != 0 else probabilities

# Path to your JSON file
json_file_path = r'LTU_Final_API\IEMOCAP\Logs\prediction_log_single_second_run.json'
# Path to the ground truth JSON file
ground_truth_file_path = r'LTU_Final_API\IEMOCAP\iemocap_ambiguous.json'

# Load the JSON data from the file
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

# Load the ground truth data from the file
with open(ground_truth_file_path, 'r') as f:
    ground_truth_data = json.load(f)

# Correct token mapping for the emotions
emotion_tokens = {
    'Anger': 'Ang er',
    'Happiness': 'H app iness',
    'Sadness': 'Sad ness',
    'Neutral': 'Ne ut ral'
}

# Initialize dictionaries to store probabilities
audio_file_softmax_sum = defaultdict(lambda: {'Anger': [], 'Happiness': [], 'Sadness': [], 'Neutral': []})

# Iterate over each entry in the JSON data
for entry in json_data:
    audio_path = entry.get('audio_id', None)
    
    if not audio_path:
        continue  # Skip if no audio path is found
    
    audio_identifier = extract_audio_id_from_path(audio_path)
    if not audio_identifier:
        continue  # Skip if audio ID extraction fails
    
    steps = entry.get('steps', [])
    
    # Process each step
    for step in steps:
        predicted_token = step.get('predicted_token', '').strip()
        logits = step.get('logits_for_target_words', {})

        # Check if the predicted token corresponds to one of the emotions
        for emotion, token in emotion_tokens.items():
            if predicted_token.startswith(token.split()[0]):  # Match the first part of the token
                # Check if logits are available for each emotion, use default if not
                anger_logits = np.mean(logits['Anger']) if 'Anger' in logits else 0
                happiness_logits = np.mean(logits['Happiness']) if 'Happiness' in logits else 0
                sadness_logits = np.mean(logits['Sadness']) if 'Sadness' in logits else 0
                neutral_logits = np.mean(logits['Neutral']) if 'Neutral' in logits else 0

                # Extract the logits for all emotions
                emotion_logits = [anger_logits, happiness_logits, sadness_logits, neutral_logits]

                # Apply softmax to the logits
                softmax_probs = np.exp(emotion_logits) / np.sum(np.exp(emotion_logits))

                # Store the softmax probability for the current emotion
                audio_file_softmax_sum[audio_identifier]['Anger'].append(softmax_probs[0])
                audio_file_softmax_sum[audio_identifier]['Happiness'].append(softmax_probs[1])
                audio_file_softmax_sum[audio_identifier]['Sadness'].append(softmax_probs[2])
                audio_file_softmax_sum[audio_identifier]['Neutral'].append(softmax_probs[3])

# Now, average the softmax probabilities for each audio file and normalize them
audio_file_final_probs = {}
audio_files_with_softmax_count = 0
audio_files_with_single_label_count = 0

# Initialize lists for ground truth and predicted labels
ground_truth_labels = []
predicted_labels = []

for audio_identifier, emotion_probs in audio_file_softmax_sum.items():
    # Compute the average softmax probabilities for each emotion
    avg_anger = np.mean(emotion_probs['Anger']) if emotion_probs['Anger'] else 0
    avg_happiness = np.mean(emotion_probs['Happiness']) if emotion_probs['Happiness'] else 0
    avg_sadness = np.mean(emotion_probs['Sadness']) if emotion_probs['Sadness'] else 0
    avg_neutral = np.mean(emotion_probs['Neutral']) if emotion_probs['Neutral'] else 0

    # Normalize the averaged probabilities
    normalized_probs = normalize([avg_anger, avg_happiness, avg_sadness, avg_neutral])

    # Store the normalized probabilities for the audio file
    audio_file_final_probs[audio_identifier] = {
        'Anger': normalized_probs[0],
        'Happiness': normalized_probs[1],
        'Sadness': normalized_probs[2],
        'Neutral': normalized_probs[3]
    }

    # Find the most probable emotion
    emotions = ['Anger', 'Happiness', 'Sadness', 'Neutral']
    most_probable_emotion = emotions[np.argmax(normalized_probs)]  # Get the emotion with the highest probability

    # Compare with ground truth if it requires prediction
    ground_truth_entry = next((item for item in ground_truth_data if item['id'] == audio_identifier), None)
    if ground_truth_entry and ground_truth_entry['need_prediction'] == 'yes' and len(set(ground_truth_entry['emotion'])) == 1:
        ground_truth_emotion = ground_truth_entry['emotion'][0]
        ground_truth_labels.append(ground_truth_emotion)
        predicted_labels.append(most_probable_emotion)
        print(f"Audio ID: {audio_identifier}")
        print(f"Ground truth emotion: {ground_truth_emotion}")
        print(f"Predicted most probable emotion: {most_probable_emotion}")
        print(f"Anger: {normalized_probs[0]:.4f}, Happiness: {normalized_probs[1]:.4f}, Sadness: {normalized_probs[2]:.4f}, Neutral: {normalized_probs[3]:.4f}\n")
        audio_files_with_single_label_count += 1

    # Increment count for audio files with calculated softmax probabilities
    audio_files_with_softmax_count += 1

# Output the total count of audio files with softmax probabilities
print(f"Total number of audio files with softmax probabilities calculated: {audio_files_with_softmax_count}")
# Output the total count of audio files with a single ground truth label and needing prediction
print(f"Total number of audio files with a single ground truth label and needing prediction: {audio_files_with_single_label_count}")

# Calculate metrics if there are any ground truth and predicted labels
if ground_truth_labels and predicted_labels:
    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    
    # Calculate confusion matrix
    labels = ['Anger', 'Happiness', 'Sadness', 'Neutral']
    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels, labels=labels)
    
    # Calculate precision, recall, and F1 score (unweighted)
    precision_unweighted = precision_score(ground_truth_labels, predicted_labels, average='macro')
    recall_unweighted = recall_score(ground_truth_labels, predicted_labels, average='macro')
    f1_unweighted = f1_score(ground_truth_labels, predicted_labels, average='macro')
    
    # Calculate precision, recall, and F1 score (weighted)
    precision_weighted = precision_score(ground_truth_labels, predicted_labels, average='weighted')
    recall_weighted = recall_score(ground_truth_labels, predicted_labels, average='weighted')
    f1_weighted = f1_score(ground_truth_labels, predicted_labels, average='weighted')

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
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()