from collections import defaultdict
import re
import numpy as np
# from dictances import bhattacharyya_coefficient, kullback_leibler
from dictances import kullback_leibler
# from sklearn.metrics import r2_score


# Small epsilon value to prevent log(0) issues
EPSILON = 1e-10

def bhattacharyya_coefficient(p, q):

    """
    Calculate the Bhattacharyya Coefficient between two probability distributions.
    Parameters:
    p (array-like): First probability distribution (must sum to 1).
    q (array-like): Second probability distribution (must sum to 1)
    Returns:
    float: Bhattacharyya Coefficient.
    """

    # Ensure both distributions are numpy arrays
    p = np.asarray(p)
    q = np.asarray(q)
    # Compute the Bhattacharyya Coefficient
    bc = np.sum(np.sqrt(p * q))
    return bc

def R(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    epsilon = EPSILON
    covariance = np.cov(a,b, ddof=0)[0][1]

    var_a = np.var(a, ddof=0)
    var_b = np.var(b, ddof=0)
    R_square = round(covariance**2 / (var_a * var_b + epsilon), 4)
    return R_square

def normalize_emotion_name(emotion):
    return "Neutral" if emotion.lower() in ["neutral state", "neutral"] else emotion

def extract_predictions(log_data):
    results = {}
    emotion_map = {"Anger": 0, "Happiness": 1, "Sadness": 2, "Neutral": 3}
    non_standard_count = 0  # Counter for non-standard predictions

    for block in log_data.strip().split('----------------------------------------'):
        id_match = re.search(r'ID: (.+)', block)
        prediction_match = re.search(r'Prediction: (.+)', block)
        emotions_match = re.search(r'Emotions: (.+)', block)
        
        if id_match and prediction_match and emotions_match:
            entry_id = id_match.group(1).strip()

            prediction_str = prediction_match.group(1)
            emotions_str = emotions_match.group(1)

            # Check for non-standard predictions (e.g., missing digits)
            if not any(char.isdigit() for char in prediction_str):
                non_standard_count += 1
                continue

            # Extract prediction values
            prediction_dict = {}
            total_percentage = 0
            for pred in prediction_str.split(','):
                if ':' in pred:
                    emotion, value = pred.split(':', 1)
                    emotion = normalize_emotion_name(emotion.strip())
                    value = re.search(r'\d+', value.strip())
                    if value:
                        value = int(value.group())
                        prediction_dict[emotion] = value
                        total_percentage += value

            # Normalize prediction values
            if total_percentage > 0:
                normalized_prediction = {emotion: prediction_dict.get(emotion, 0) / total_percentage
                                         for emotion in emotion_map}  # Normalize to sum 1
            else:
                normalized_prediction = {emotion: 0 for emotion in emotion_map}

            # Extract emotion distribution (ground truth)
            emotion_list = [normalize_emotion_name(label.strip()) for label in emotions_str.split(',')]
            emotion_count = len(emotion_list)
            emotion_distribution = defaultdict(int)

            for label in emotion_list:
                emotion_distribution[label] += 1

            # Normalize emotion distribution to sum to 1 (probability distribution)
            normalized_emotion_distribution = {emotion: emotion_distribution.get(emotion, 0) / emotion_count
                                               for emotion in emotion_map}

            # Group prediction and ground truth together
            results[entry_id] = {
                "prediction": normalized_prediction,
                "ground_truth": normalized_emotion_distribution
            }

    return results, non_standard_count

# Step 1: Read the log data from the file
with open('C:/Users/julev/Desktop/ltuAPI/LTU_Final_API/IEMOCAP/prediction_log_final.txt', 'r') as file:
    log_data = file.read()

# Step 2: Call the function with the log data
results, non_standard_count = extract_predictions(log_data)

total_bc = 0
total_kl = 0
total_r2 = 0
count = 0

# Step 3: Perform metrics calculations in the loop
for entry_id, data in results.items():
    prediction = data['prediction']
    ground_truth = data['ground_truth']

    # Convert the prediction and ground truth to arrays for calculations
    pred_values = np.array([prediction[emotion] for emotion in prediction]) + EPSILON
    actual_values = np.array([ground_truth[emotion] for emotion in ground_truth]) + EPSILON

    # Normalize the prediction and actual values
    pred_values /= pred_values.sum()  # Normalize to sum to 1
    actual_values /= actual_values.sum()  # Normalize to sum to 1
    count += 1

    # 1. Bhattacharyya Coefficient
    bhattacharyya = bhattacharyya_coefficient(pred_values, actual_values)
    total_bc += bhattacharyya
    
    # 2. Kullback-Leibler Divergence 
    kl_divergence = kullback_leibler(
        {key: value + EPSILON for key, value in ground_truth.items()},
        {key: value + EPSILON for key, value in prediction.items()}
    )
    total_kl += kl_divergence

    # 3. R-squared
    r_squared = R(actual_values, pred_values)
    total_r2 += r_squared

final_bc = total_bc / count
final_kl = total_kl / count
final_r2 = total_r2 / count

print(f"  Bhattacharyya Coefficient: {final_bc:.4f}")
print(f"  Kullback-Leibler Divergence: {final_kl:.4f}")
print(f"  R-squared: {final_r2:.4f}\n")

# Display the number of non-standard outputs
print(f"Number of non-standard outputs: {non_standard_count}")
