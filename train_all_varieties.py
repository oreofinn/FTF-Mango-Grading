import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

# --- Config ---
DATA_DIR = "mango_dataset"    # must contain subfolders: "APPLE MANGO", "CARABAO MANGO", etc.
IMG_SIZE  = (100, 100)
BATCH     = 32
EPOCHS    = 10

# common augmenter + split
augmenter = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.8,1.2],
    horizontal_flip=True
)

for variety in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, variety)
    if not os.path.isdir(path):
        continue

    print(f"\n??  Training variety: {variety}")
    train_gen = augmenter.flow_from_directory(
        path,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        subset="training"
    )
    val_gen = augmenter.flow_from_directory(
        path,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = train_gen.num_classes

    # build a simple CNN
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, (3,3), activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128,(3,3), activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    # save under a safe filesystem name
    safe = variety.lower().replace(" ","_")
    outname = f"cnn_{safe}.keras"
    model.save(outname)
    print(f"? Saved model for {variety} ? `{outname}`")
