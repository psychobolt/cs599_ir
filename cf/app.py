import os
import json
import math
from typing import List, Dict


PATH = os.path.dirname(__file__)
OUTPUT_DIR = f'{PATH}/output'
TEST_DATA_FILEPATH = f'{PATH}/data/test.json'
TRAINING_DATA_FILEPATH = f'{PATH}/data/training.json'
PREDICTIONS_DATA_FILEPATH = f'{OUTPUT_DIR}/predictions.json'
EVAL_DATA_FILEPATH = f'{OUTPUT_DIR}/eval.json'
Vector = List[float]


def dot(A: Vector, B: Vector) -> float:
    """
    Returns the Dot Product between two vectors
    """
    if (len(A) != len(B)):
        return 0
    return sum(ab[0] * ab[1] for ab in zip(A, B))


def norm(A: Vector) -> float:
    """
    Returns the Norm between two vectors
    """
    return math.sqrt(sum(a * a for a in A))


def cosSim(A: Vector, B: Vector) -> float:
    """
    Returns the Cosine Similarity between two vectors
    """
    if (len(A) != len(B)):
        return 0
    return dot(A, B) / (norm(A) * norm(B))


def avg(items: List, titles: List[str]) -> float:
    """
    Returns the average rating for each title
    """
    total = 0
    n = 0
    for item in items:
        if (item['title'] in titles):
            total += item['rating']
            n += 1
    if n == 0:
        return 0
    return total / n


def pearson(
        user: Dict,
        category: str,
        items: List[Dict],
        neighbors: List[Dict],
        avgRatings={}) -> Dict:
    """
    Returns the Pearson Correlations between a user specified category-item
    ratings and its neighbor's
    """
    userId = user['id'] or user['name']
    titles = [item['title'] for item in items]
    if (userId in avgRatings):
        avgRating = avgRatings[userId]
    else:
        avgRating = avg(items, titles)
        avgRatings[userId] = avgRating
    vecA = [item['rating'] - avgRating for item in items]
    correlations = {}

    for neighbors in neighbors:
        items = neighbors['ratings'][category]
        if items:
            neighborId = neighbors['id'] or neighbors['name']
            if (neighborId in avgRatings):
                avgRating = avgRatings[neighborId]
            else:
                avgRating = avg(items, titles)
                avgRatings[neighborId] = avgRating
            vecB = [item['rating'] -
                    avgRating for item in items if item['title'] in titles]
            correlations[neighborId] = cosSim(vecA, vecB)
    return correlations


def getRating(items: Dict, title: str):
    """
    Get rating for specific item
    """
    for data in items:
        if data['title'] == item['title']:
            return data['rating']
    return None

# Create empty output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the training and test data

with open(TEST_DATA_FILEPATH, 'r') as testFile:
    TEST_DATA = json.load(testFile)

with open(TRAINING_DATA_FILEPATH, 'r') as trainingFile:
    TRAINING_DATA = json.load(trainingFile)

results = []
for user in TEST_DATA:
    ratings = user['ratings']
    observed = user['observed']
    userId = user['id'] or user['name']
    correlations = {}
    predictions = {}

    for category, items in ratings.items():
        # Compute the Pearson Correlations
        avgRatings = {}
        correlations[category] = pearson(user, category, items,
                                         TRAINING_DATA, avgRatings)

    for category, items in observed.items():
        # Predict user ratings from observed test data using correlations
        prediction = []
        predictions[category] = prediction

        for item in items:
            title = item['title']
            numerator = 0
            for neighbor in TRAINING_DATA:
                neighborId = neighbor['id'] or neighbor['name']
                numerator += correlations[category][neighborId] * (
                    getRating(neighbor['ratings'][category], title) - avgRatings[neighborId])
            prediction.append({
                'title': title,
                'rating': numerator / sum(abs(v) for v in correlations[category].values()) + avgRatings[userId]
            })

    results.append({
        'id': user['id'],
        'name': user['name'],
        'predictions': predictions
    })

# Evaluate the overall prediction performance
mae = 0
rsme = 0
count = 0
summation = 0
sumproduct = 0
for i, user in enumerate(TEST_DATA):
    for category, items in user['observed'].items():
        count += len(items)
        for item in items:
            error = abs(item['rating'] - getRating(results[i]
                                                   ['predictions'][category], item['title']))
            summation += error
            sumproduct += error ** 2

mae = summation / count
rmse = math.sqrt(sumproduct / count)


# Store predictions
with open(PREDICTIONS_DATA_FILEPATH, 'w+') as predictionsFile:
    json.dump(results, predictionsFile, indent=2)

print("""\

Ratings have been predicted. Please see output/predictions.json

Performance:
        MAE: {}
       RMSE: {}
""".format(mae, rmse))
