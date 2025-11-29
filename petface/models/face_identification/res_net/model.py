import keras
import tensorflow as tf

from typing import Literal

from petface.const import IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH


ModelLoss = Literal["softmax", "arcface"]


class ArcFaceLayer(keras.layers.Layer):
    def __init__(self, num_classes: int, *, margin: float = 0.05, scale: int = 20, **kwargs):
        self._num_classes = num_classes
        self._margin = margin
        self._scale = scale
        super().__init__(**kwargs)

    def build(self, input_shape):
        embedding_dim = input_shape[-1]

        self.W = self.add_weight(
            name="W",
            shape=(embedding_dim, self._num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )
    
    def call(self, inputs: tf.Tensor, labels: tf.Tensor):
        embeddings = tf.nn.l2_normalize(inputs, axis=1)
        weights = tf.nn.l2_normalize(self.W, axis=0)

        logits = tf.matmul(embeddings, weights)

        theta = tf.acos(tf.clip_by_value(logits, -1 + 1e-7, 1 - 1e-7))

        cos_theta_m = tf.cos(theta + self._margin)

        one_hot = tf.one_hot(labels, depth=self._num_classes)

        final_logits = logits * (1 - one_hot) + cos_theta_m * one_hot

        return final_logits * self._scale


def get_backbone_model(num_classes: int):
    backbone = keras.applications.ResNet50(
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH),
        pooling="avg",
        classes=num_classes,
        weights="imagenet",
    )

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-30:]:
        layer.trainable = True

    return backbone


def get_embedding_model(num_classes: int):
    backbone = get_backbone_model(num_classes)

    embedding_model = keras.models.Sequential()
    embedding_model.add(backbone)
    embedding_model.add(keras.layers.Flatten())
    embedding_model.add(keras.layers.Dense(512, use_bias=False))

    return embedding_model


def get_model(
    num_classes: int,
    model_loss: ModelLoss
):
    embedding_model = get_embedding_model(num_classes)

    if model_loss == "softmax":
        embedding_model.add(keras.layers.BatchNormalization(name="embedding"))
        embedding_model.add(keras.layers.Dense(num_classes))
    elif model_loss == "arcface":
        img_inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), name="images")
        label_inputs = keras.Input(shape=(), dtype=tf.int32, name="labels")

        embeddings = embedding_model(img_inputs)

        logits = ArcFaceLayer(num_classes)(embeddings, label_inputs)

        embedding_model = keras.Model(inputs=[img_inputs, label_inputs], outputs=logits)
    else:
        raise NotImplementedError(model_loss)

    embedding_model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=50, name="top50")
        ]
    )

    return embedding_model
