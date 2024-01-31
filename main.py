import random
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Load dataset
ds, ds_info = tfds.load('food101', shuffle_files=True, as_supervised=True, with_info=True)
train_ds, valid_ds = ds["train"], ds["validation"]

# Constants
MAX_SIDE_LEN = 128
HOT_DOG_CLASS = 55
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001

# Preprocessing function
def preprocess(image, label):
    image = tf.cast(tf.image.resize(image, [MAX_SIDE_LEN, MAX_SIDE_LEN]), dtype=tf.int32)
    label = tf.cast(label == HOT_DOG_CLASS, dtype=tf.int32)
    return image, label

# Apply preprocessing
train_ds = train_ds.map(preprocess)
valid_ds = valid_ds.map(preprocess)

# Filter dataset for hotdogs and not hotdogs
train_hotdogs = train_ds.filter(lambda _, label: label == 1).repeat(3)
train_nothotdogs = train_ds.filter(lambda _, label: label == 0)
valid_hotdogs = valid_ds.filter(lambda _, label: label == 1).repeat(3)
valid_nothotdogs = valid_ds.filter(lambda _, label: label == 0)

# Combining datasets
train_ds = tf.data.Dataset.sample_from_datasets([train_hotdogs, train_nothotdogs], weights=[0.5, 0.5], stop_on_empty_dataset=True)
train_ds = train_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

valid_ds = tf.data.Dataset.sample_from_datasets([valid_hotdogs, valid_nothotdogs], weights=[0.5, 0.5], stop_on_empty_dataset=True)
valid_ds = valid_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Set random seed
random.seed(0)

# Model definition
model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(MAX_SIDE_LEN, MAX_SIDE_LEN, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1)
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1
)

# Save model
model.save('hotdog.h5')
