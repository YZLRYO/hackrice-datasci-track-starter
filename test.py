import numpy as np
import pandas as pd

from models import siamese_model, simple_tower
from artist_dataset import ArtistDataset, TestDatasetGen

# Random Seed (for reproducibility)
SEED = 2525
np.random.seed(SEED)
# Directory information
PATH_TO_TEST = "../data/test/"
PATH_TO_SUBMISSION_INFO = "../data/submission_info.csv"
# Dataset settings
IMG_SIZE = (128, 128)
# Model settings
TOWER_FUNC = simple_tower
MODEL_FUNC = siamese_model
# Training settings
BATCH_SIZE = 32


test_dataset = ArtistDataset(PATH_TO_TEST, img_size=IMG_SIZE)

print("Test Dataset: %s samples" % len(test_dataset))

# Create the generator for the test set
testgen = TestDatasetGen(
    test_dataset, PATH_TO_SUBMISSION_INFO, batch_size=BATCH_SIZE)

# Initialze the model
model = MODEL_FUNC(TOWER_FUNC(), IMG_SIZE)
# Load the model's weights
model.load_weights(MODEL_FUNC.__name__ + ".h5")

# Make the predictions
preds = model.predict_generator(testgen, testgen.steps_per_epoch, verbose=1)
print(preds.shape)

# Create the submission
print("Creating Submission")
submission = pd.DataFrame(np.hstack(
    [np.arange(len(preds), dtype=int).reshape(-1, 1), preds]),
    columns=['index', 'sameArtist'])
submission.to_csv("../submission.csv", index=False)
