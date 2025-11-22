import pathlib
import keras
import tensorflow as tf


ANIMAL = "guineapig"
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

num_classes = len(train_ds.class_names)  # type: ignore

AUTOTUNE = tf.data.AUTOTUNE

num_classes = len(train_ds.class_names)  # type: ignore

train_ds = train_ds.prefetch(AUTOTUNE)  # type: ignore
val_ds = val_ds.prefetch(AUTOTUNE)  # type: ignore

backbone = keras.applications.ResNet50(
    include_top=False,
    input_shape=(img_height, img_width, 3),
    pooling="avg",
    classes=num_classes,
    weights="imagenet",
)

for layer in backbone.layers:
    layer.trainable = False


model = keras.models.Sequential()

# TODO: Try arcface loss
model.add(backbone)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(num_classes))

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

checkpoint_path = CHECKPOINTS_DIR / "training_3/cp.weights.h5"

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=25,
  callbacks=[cp_callback, early_stop]
)
