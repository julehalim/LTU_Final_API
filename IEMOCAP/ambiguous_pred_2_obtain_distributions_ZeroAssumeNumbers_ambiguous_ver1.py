import sys
import json
import numpy as np
import re
from collections import defaultdict, Counter
from dictances import kullback_leibler

# Small epsilon value to prevent log(0) or divide-by-zero issues
EPSILON = 1e-10

# Ensure UTF-8 encoding is used
sys.stdout.reconfigure(encoding='utf-8')

# Function to normalize probabilities (so they sum to 1)
def normalize(probabilities):
    total = sum(probabilities)
    return [p / total for p in probabilities] if total != 0 else probabilities

# Softmax function for logits
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Subtracting max for numerical stability
    return exp_logits / exp_logits.sum()

# Function to extract the audio ID from the audio identifier
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

# Bhattacharyya Coefficient calculation
def bhattacharyya_coefficient(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    bc = np.sum(np.sqrt(p * q))
    return bc

# R-squared calculation
def R(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    epsilon = EPSILON
    covariance = np.cov(a, b, ddof=0)[0][1]

    var_a = np.var(a, ddof=0)
    var_b = np.var(b, ddof=0)
    R_square = round(covariance ** 2 / (var_a * var_b + epsilon), 4)
    return R_square

# Function to calculate the emotion distribution from the ground truth
def get_emotion_distribution(emotions):
    emotion_count = Counter(emotions)
    total = len(emotions)
    distribution = {
        'Anger': 0,
        'Happiness': 0,
        'Sadness': 0,
        'Neutral': 0
    }
    for emotion in emotion_count:
        if 'Anger' in emotion:
            distribution['Anger'] = (emotion_count[emotion] / total) * 100
        elif 'Happiness' in emotion:
            distribution['Happiness'] = (emotion_count[emotion] / total) * 100
        elif 'Sadness' in emotion:
            distribution['Sadness'] = (emotion_count[emotion] / total) * 100
        elif 'Neutral' in emotion:
            distribution['Neutral'] = (emotion_count[emotion] / total) * 100
    return distribution

# Path to your JSON files
json_file_path = r'LTU_Final_API\IEMOCAP\Logs\prediction_log_single_second_run.json'
groundtruth_file_path = r'LTU_Final_API\IEMOCAP\iemocap_ambiguous.json'

# Load the JSON data from the file
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

with open(groundtruth_file_path, 'r') as gt_file:
    groundtruth_data = json.load(gt_file)

# Correct token mapping for the emotions
emotion_tokens = {
    'Anger': 'Ang er',
    'Happiness': 'H app iness',
    'Sadness': 'Sad ness',
    'Neutral': 'Ne ut ral'
}

# Initialize dictionaries to store probabilities
audio_file_softmax_sum = defaultdict(lambda: {'Anger': [], 'Happiness': [], 'Sadness': [], 'Neutral': []})
final_emotion_outputs = defaultdict(dict)

# Initialize a count for how many calculations are done
calculation_count = 0

# Initialize variables to accumulate metric scores
# Initialize variables to accumulate metric scores and track comparisons
total_bc = 0
total_kl = 0
total_r2 = 0
count = 0  # This will track the number of files compared (ground truth vs prediction)

# Iterate over each entry in the JSON data
for entry in json_data:
    audio_identifier = entry.get('audio_id', None)
    
    if not audio_identifier:
        continue  # Skip if no audio identifier is found
    
    # Extract the ID using the provided function
    audio_id = extract_audio_id_from_path(audio_identifier)
    if not audio_id:
        print(f"Could not extract ID from: {audio_identifier}")
        continue
    
    steps = entry.get('steps', [])
    current_emotion = None  # Track which emotion label is being generated
    current_number = ''     # Track the number that follows the emotion
    number_logits = []      # Accumulate logits for the number digits
    
    # Process each step
    for step in steps:
        predicted_token = step.get('predicted_token', '').strip()
        logits = step.get('logits_for_target_words', {})

        # Check if the predicted token corresponds to an emotion label
        for emotion, token in emotion_tokens.items():
            if predicted_token.startswith(token.split()[0]):  # Match the first part of the token
                current_emotion = emotion  # Start tracking the current emotion being generated
                number_logits = []  # Reset number logits when a new emotion is detected
                break
        
        # If a number is being generated, track it and collect the logits
        if current_emotion and predicted_token.isdigit():
            current_number += predicted_token  # Concatenate the digit to form the full number
            
            # Accumulate logits for the digits
            number_logits.append([
                np.mean(logits['Anger']) if logits['Anger'] else 0,
                np.mean(logits['Happiness']) if logits['Happiness'] else 0,
                np.mean(logits['Sadness']) if logits['Sadness'] else 0,
                np.mean(logits['Neutral']) if logits['Neutral'] else 0
            ])
        
        # Handle percentage or other related tokens
        if current_emotion and predicted_token == '%':
            # Combine the logits for the emotion and the accumulated number logits
            emotion_logits = [
                np.mean(logits['Anger']) if logits['Anger'] else 0,
                np.mean(logits['Happiness']) if logits['Happiness'] else 0,
                np.mean(logits['Sadness']) if logits['Sadness'] else 0,
                np.mean(logits['Neutral']) if logits['Neutral'] else 0
            ]
            
            # Sum the emotion logits with the number logits for each emotion
            combined_logits = np.sum([emotion_logits] + number_logits, axis=0)

            # Apply softmax to the combined logits
            softmax_probs = softmax(combined_logits)

            # Store the softmax probabilities for the generated number
            audio_file_softmax_sum[audio_identifier][current_emotion].append(softmax_probs)

            # Store the final emotion and its associated percentage
            final_emotion_outputs[audio_identifier][current_emotion] = int(current_number)
            current_emotion = None  # Reset after storing the number for this emotion
            current_number = ''      # Reset the number tracker

# Now, average the softmax probabilities for each audio file and normalize them
audio_file_final_probs = {}

for audio_identifier, emotion_probs in audio_file_softmax_sum.items():
    audio_identifier = extract_audio_id_from_path(audio_identifier)
    # Compute the average softmax probabilities for each emotion, checking for empty lists
    avg_anger = np.mean([np.mean(p[0]) for p in emotion_probs['Anger']]) if emotion_probs['Anger'] else 0
    avg_happiness = np.mean([np.mean(p[1]) for p in emotion_probs['Happiness']]) if emotion_probs['Happiness'] else 0
    avg_sadness = np.mean([np.mean(p[2]) for p in emotion_probs['Sadness']]) if emotion_probs['Sadness'] else 0
    avg_neutral = np.mean([np.mean(p[3]) for p in emotion_probs['Neutral']]) if emotion_probs['Neutral'] else 0

    # Normalize the averaged probabilities so they sum to 1
    normalized_probs = normalize([avg_anger, avg_happiness, avg_sadness, avg_neutral])

    # Store the normalized probabilities for the audio file
    audio_file_final_probs[audio_identifier] = {
        'Anger': normalized_probs[0],
        'Happiness': normalized_probs[1],
        'Sadness': normalized_probs[2],
        'Neutral': normalized_probs[3]
    }

    # Get ground truth for the corresponding audio file
    groundtruth_entry = next((item for item in groundtruth_data if extract_audio_id_from_path(item['audio']) == audio_identifier and item.get('need_prediction', '').lower() == 'yes'), None)
    
    if not groundtruth_entry:
        print(f"No matching ground truth found for {audio_identifier} with need_prediction = 'yes'")
        continue
    
    # Get the ground truth emotion distribution
    groundtruth_emotion_distribution = get_emotion_distribution(groundtruth_entry['emotion'])

    # Convert prediction and ground truth to arrays for calculations
    prediction = np.array([normalized_probs[i] for i in range(4)]) + EPSILON
    ground_truth = np.array([groundtruth_emotion_distribution[emotion] / 100 for emotion in emotion_tokens]) + EPSILON

    # Normalize the prediction and ground truth arrays
    prediction /= prediction.sum()
    ground_truth /= ground_truth.sum()

    # Calculate Bhattacharyya Coefficient
    bhattacharyya = bhattacharyya_coefficient(prediction, ground_truth)
    total_bc += bhattacharyya

    # Calculate Kullback-Leibler Divergence
    kl_divergence = kullback_leibler(
        {key: value + EPSILON for key, value in zip(emotion_tokens, ground_truth)},
        {key: value + EPSILON for key, value in zip(emotion_tokens, prediction)}
    )
    total_kl += kl_divergence

    # Calculate R-squared
    r_squared = R(ground_truth, prediction)
    total_r2 += r_squared

    count += 1  # Increment the comparison count for each matched file

    # Output predicted and ground truth for each file
    print(f"File ID: {audio_identifier}")
    print(f"Predicted: {audio_file_final_probs[audio_identifier]}")
    print(f"Ground Truth: {groundtruth_emotion_distribution}")
    print("---")

# Calculate final metrics averages
final_bc = total_bc / count if count > 0 else 0
final_kl = total_kl / count if count > 0 else 0
final_r2 = total_r2 / count if count > 0 else 0

# Output the final metrics
print(f"Bhattacharyya Coefficient: {final_bc:.4f}")
print(f"Kullback-Leibler Divergence: {final_kl:.4f}")
print(f"R-squared: {final_r2:.4f}")

# Output the total number of ground truth vs prediction comparisons made
print(f"Total number of ground truth vs prediction comparisons made: {count}")
