import keras
import tensorflow as tf
import pathlib


ANIMAL = "chimp"
CHECKPOINTS_DIR = pathlib.Path(__file__).parent.parent / "checkpoints" / ANIMAL / "convolutional_net"

if not CHECKPOINTS_DIR.exists():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

img_height, img_width = 224, 224
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    directory=f"/home/mtofano/src/en-625-638-group-project/dataset/out/images/train/{ANIMAL}",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    directory=f"/home/mtofano/src/en-625-638-group-project/dataset/out/images/train/{ANIMAL}",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

AUTOTUNE = tf.data.AUTOTUNE

num_classes = len(train_ds.class_names)  # type: ignore

train_ds = train_ds.prefetch(AUTOTUNE)  # type: ignore
val_ds = val_ds.prefetch(AUTOTUNE)  # type: ignore

model = keras.models.Sequential(
    layers=[
        # Rescale each pixel to be between 0 and 1
        keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        keras.layers.Conv2D(32, 3, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name="embedding"),
        keras.layers.Dense(num_classes)
    ]
)

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=50, name="top50")
    ]
)

print(model.summary())

checkpoint_path = CHECKPOINTS_DIR / "training_2/cp.weights.h5"

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=50,
  callbacks=[cp_callback, early_stop]
)
