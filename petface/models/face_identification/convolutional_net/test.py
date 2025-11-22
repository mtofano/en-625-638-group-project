import keras
import tensorflow as tf
import pathlib
import numpy as np


ANIMAL = "guineapig"
CHECKPOINT_PATH = pathlib.Path(__file__).parent.parent / "checkpoints" / ANIMAL / "convolutional_net" / "training_1" / "cp.weights.h5"

img_height, img_width = 224, 224
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    directory=f"/home/mtofano/src/en-625-638-group-project/dataset/out/images/train/{ANIMAL}",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    directory=f"/home/mtofano/src/en-625-638-group-project/dataset/out/images/test/{ANIMAL}",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

num_classes = len(train_ds.class_names)  # type: ignore

model = keras.models.Sequential(
    layers=[
        # Rescale each pixel to be between 0 and 1
        keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu', use_bias=False),
        # Normalize across each filter to follow ~ N(0, 1)
        keras.layers.BatchNormalization(),
        # Normalize each vector to unit distance
        keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1)),
        keras.layers.Dense(num_classes)
    ]
)

# Load weights into the imported model
model.load_weights(CHECKPOINT_PATH)

# Create embedding model (without final Dense layer)
embedding_model = keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output  # Get output before final Dense layer
)

gallery_embeddings = embedding_model.predict(train_ds)

gallery_labels = tf.concat([labels for _, labels in train_ds], axis=0).numpy()  # type: ignore

all_predictions = []
all_true_labels = []

for images, labels in test_ds:
    test_batch_embeddings = embedding_model.predict(images)
    
    similiarities = np.dot(gallery_embeddings, np.transpose(test_batch_embeddings))

    predicted_label_indexes = np.argsort(np.transpose(similiarities), axis=1)[:, ::-1][:, 0]
    
    predicted_labels = gallery_labels[predicted_label_indexes]
    
    all_predictions.extend(predicted_labels)
    all_true_labels.extend(labels.numpy())

# Calculate accuracy
accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Correct predictions: {np.sum(np.array(all_predictions) == np.array(all_true_labels))} / {len(all_true_labels)}")
