import numpy as np
from keras.callbacks import ModelCheckpoint

from models import siamese_model, simple_tower
from artist_dataset import ArtistDataset, DoubleInputGen

# Random Seed (for reproducibility)
SEED = 2525
np.random.seed(SEED)
# Directory information
PATH_TO_TRAIN = "../train/"
PATH_TO_TRAIN_INFO = "../train_info.csv"
# Dataset settings
IMG_SIZE = (128, 128)
VAL_SPLIT = 0.2
SHUFFLE_SPLIT = True
# Model settings
TOWER_FUNC = simple_tower
MODEL_FUNC = siamese_model
# Training settings
BATCH_SIZE = 32
SHUFFLE_GEN = True
EPOCHS = 15


dataset = ArtistDataset(PATH_TO_TRAIN, PATH_TO_TRAIN_INFO, img_size=IMG_SIZE)
# Split the dataset into a training and validation set
train_dataset, val_dataset = dataset.validation_split(
    split=0.2, shuffle=SHUFFLE_SPLIT, seed=np.random.randint(10000))

print("Train Dataset: %s samples" % len(train_dataset))
print("Validation Dataset: %s samples" % len(val_dataset))

# Create the generators for the batches
traingen = DoubleInputGen(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=SHUFFLE_GEN, seed=np.random.randing(10000))
valgen = DoubleInputGen(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=SHUFFLE_GEN, seed=np.random.randing(10000))

# Initialize the model
model = MODEL_FUNC(TOWER_FUNC(), IMG_SIZE)

# Set up any callbacks
# This will save the best model by loss on the validation set as we train
best_model = ModelCheckpoint(model.__name__ + ".h5", monitor='val_loss',
                             verbose=1, save_best_only=True,
                             save_weights_only=True)
callbacks = [best_model]

# Train the model
model.fit_generator(traingen, traingen.steps_per_epoch,
                    epochs=EPOCHS, callbacks=callbacks, validation_data=valgen,
                    validation_steps=valgen.steps_per_epoch)
