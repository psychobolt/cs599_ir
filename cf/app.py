import os
import json
import math
from typing import List, Dict


PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = f'{PATH}/output'
TEST_DATA_FILEPATH = f'{PATH}/data/test.json'
TRAINING_DATA_FILEPATH = f'{PATH}/data/training.json'
PREDICTIONS_DATA_FILEPATH = f'{OUTPUT_DIR}/prediction.json'
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
    Returns the Norm of a vector
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


def pearson(testVec: Vector, category, neighbor, avgRatings):
    """
    Compute the Pearson Correlation using each category-item ratings 
    from the test user and its neighbor respectively
    """
    vecA = testVec
    items = neighbor['ratings'][category]
    if len(item) == 0:
        return None
    neighborId = neighbor['id'] or neighbor['name']
    if (neighborId in avgRatings):
        avgRating = avgRatings[neighborId]
    else:
        avgRating = avg(items, titles)
        avgRatings[neighborId] = avgRating
    vecB = [item['rating'] -
            avgRating for item in items if item['title'] in titles]
    return cosSim(vecA, vecB)


def getRating(items: Dict, title: str):
    """
    Get rating for specific item
    """
    for data in items:
        if data['title'] == item['title']:
            return data['rating']
    return None


def predict(
        userId,
        category: str,
        title: str,
        avgRatings: Dict,
        neighbors: List[Dict],
        correlations: Dict) -> float:
    """
    Returns the rating prediction for a user's category-item
    """
    numerator = 0
    for neighbor in neighbors:
        rating = getRating(neighbor['ratings'][category], title)
        if rating == None:
            return None
        neighborId = neighbor['id'] or neighbor['name']
        numerator += correlations[category][neighborId] * \
            (rating - avgRatings[neighborId])
    return numerator / sum(abs(v) for v in correlations[category].values()) \
        + avgRatings[userId]


# Create empty output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the training and test data

with open(TEST_DATA_FILEPATH, 'r') as testFile:
    TEST_DATA = json.load(testFile)

with open(TRAINING_DATA_FILEPATH, 'r') as trainingFile:
    TRAINING_DATA = json.load(trainingFile)

# Build inventories
inventories = {}
for user in TRAINING_DATA:
    ratings = user['ratings']
    for category, items in ratings.items():
        inventory = set()
        inventories[category] = inventory
        for item in items:
            inventory.add(item['title'])

# Remove title from a inventory if user did not rate it
for user in TRAINING_DATA:
    ratings = user['ratings']
    for category, items in ratings.items():
        inventory = inventories[category]
        titles = [item['title'] for item in items]
        for title in inventory.copy():
            if title not in titles:
                inventory.remove(title)

results = []
for user in TEST_DATA:
    ratings = user['ratings']
    observed = user['observed']
    userId = user['id'] or user['name']
    userName = user['name'] or user['id']
    correlations = {}
    predictions = {}

    print('\nPearson Correlation of {} and:'.format(userName))
    for category, items in ratings.items():
        items = [item for item in items if item[
            'title'] in inventories[category]]
        titles = [item['title'] for item in items]
        avgRatings = {}
        correlation = {}
        if (userId in avgRatings):
            avgRating = avgRatings[userId]
        else:
            avgRating = avg(items, titles)
            avgRatings[userId] = avgRating
        testVec = [item['rating'] - avgRating for item in items]

        # Compute the Pearson Correlation for each neighbor
        for neighbor in TRAINING_DATA:
            r = pearson(testVec, category, neighbor, avgRatings)
            # Print result
            if (r != None):
                print("{:>11} (Category: {}): {: f}".format(
                    neighbor['name'] or neighbor['id'], category, r))
            correlation[neighbor['id'] or neighbor['name']] = r
        correlations[category] = correlation
    if len(ratings.items()) == 0:
        print("None")

    for category, items in observed.items():
        # Predict user ratings from observed test data using correlations
        print('\nPredictions for {}:'.format(userName))
        prediction = []
        predictions[category] = prediction

        for item in items:
            title = item['title']
            rating = predict(userId, category, title, avgRatings,
                                  TRAINING_DATA, correlations)
            prediction.append({
                'title': title,
                'rating': rating
            })
            # Print the prediction
            print('{:>11} (Category: {}): {:f}'.format(title, category, rating))

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
with open(PREDICTIONS_DATA_FILEPATH, 'w') as predictionsFile:
    json.dump(results, predictionsFile, indent=2)

# Store evaluations
with open(EVAL_DATA_FILEPATH, 'w') as evalFile:
    json.dump({
        'mae': mae,
        'rmse': rmse
    }, evalFile, indent=2)

# Print evaluations
print("""
Performance:
        MAE: {:f}
       RMSE: {:f}
""".format(mae, rmse))
