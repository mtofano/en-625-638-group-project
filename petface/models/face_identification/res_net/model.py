import keras
import tensorflow as tf
import pathlib
import json
import plotly.graph_objects as go
import numpy as np

from abc import ABC, abstractmethod
from plotly.subplots import make_subplots
from typing import TypedDict

from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset

from petface.const import IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, ROOT_DIR


IMG_DIMENSIONS = (IMG_HEIGHT, IMG_WIDTH)


class ArcFaceLayer(keras.layers.Layer):
    def __init__(
        self, 
        num_classes: int, 
        *, 
        margin: float = 0.05, 
        scale: int = 20, 
        **kwargs
    ):
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
    

class Evaluation(TypedDict):
    top_1_accuracy: float
    top_5_accuracy: float
    top_50_accuracy: float


class ReidentModel(ABC):
    def __init__(
        self,
        out_dir: pathlib.Path,
        training_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset
    ) -> None:
        self.out_dir = out_dir
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.classification_model: keras.Model | None = None

    @property
    def number_of_classes(self) -> int:
        return len(self.training_dataset.class_names)  # type: ignore
    
    @property
    def checkpoint_file(self) -> pathlib.Path:
        checkpoint_file = self.out_dir / "checkpoints" / "cp.weights.h5"
        if not checkpoint_file.parent.exists():
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        return checkpoint_file

    @property
    def history_file(self) -> pathlib.Path:
        history_file = self.out_dir / "history" / "history.json"
        if not history_file.parent.exists():
            history_file.parent.mkdir(parents=True, exist_ok=True)
        return history_file 
    
    @abstractmethod
    def compile(self) -> None:
        ...

    @abstractmethod
    def evaluate(self, evaluation_dataset: tf.data.Dataset) -> Evaluation:
        ... 
    
    def backbone_model(self) -> keras.Model:
        backbone = keras.applications.ResNet50(
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH),
            pooling="avg",
            classes=self.number_of_classes,
            weights="imagenet",
        )

        for layer in backbone.layers:
            layer.trainable = False

        for layer in backbone.layers[-30:]:
            layer.trainable = True

        return backbone

    def process_inputs(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        training_dataset = self.training_dataset.prefetch(tf.data.AUTOTUNE)  # type: ignore
        validation_dataset = self.validation_dataset.prefetch(tf.data.AUTOTUNE)  # type: ignore

        return training_dataset, validation_dataset

    def train(self) -> None:
        if not self.classification_model:
            self.compile()

        assert self.classification_model is not None

        print(self.classification_model.summary())

        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_file,
            save_weights_only=True,
            save_best_only=True,
            verbose=1
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        training_input, validation_input = self.process_inputs()

        history = self.classification_model.fit(
            training_input,
            validation_data=validation_input,
            epochs=25,
            callbacks=[cp_callback, early_stop]
        )

        with self.history_file.open("w") as f:
            json.dump(history.history, f)

    def load_from_checkpoint(self) -> None:
        if not self.classification_model:
            self.compile()

        assert self.classification_model is not None

        self.classification_model.load_weights(self.checkpoint_file)

    def display_training_history(self) -> None:
        with self.history_file.open("r") as f:
            history = json.load(f)

        epochs = list(range(1, len(history["sparse_categorical_accuracy"])))

        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )

        fig.add_trace(
            go.Scatter(
                name="Training Accuracy",
                x=epochs,
                y=history["sparse_categorical_accuracy"],
                mode="lines"
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                name="Training Top 5 Accuracy",
                x=epochs,
                y=history["top5"],
                mode="lines"
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                name="Training Top 50 Accuracy",
                x=epochs,
                y=history["top50"],
                mode="lines"
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                name="Validation Accuracy",
                x=epochs,
                y=history["val_sparse_categorical_accuracy"],
                mode="lines"
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                name="Validation Top 5 Accuracy",
                x=epochs,
                y=history["val_top5"],
                mode="lines"
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                name="Validation Top 50 Accuracy",
                x=epochs,
                y=history["val_top50"],
                mode="lines"
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                name="Training Loss",
                x=epochs,
                y=history["loss"],
                mode="lines"
            ),
            secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                name="Validation Loss",
                x=epochs,
                y=history["val_loss"],
                mode="lines"
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Training and Validation Metrics",
            xaxis_title="Epoch"
        )

        fig.update_yaxes(
            title_text="Accuracy",
            secondary_y=False
        )

        fig.update_yaxes(
            title_text="Loss",
            secondary_y=True
        )

        fig.show()


class SoftmaxModel(ReidentModel):
    def compile(self) -> None:
        backbone_model = self.backbone_model()

        self.classification_model = keras.models.Sequential()
        self.classification_model.add(backbone_model)
        self.classification_model.add(keras.layers.Flatten())
        self.classification_model.add(keras.layers.Dense(512, use_bias=False))
        self.classification_model.add(keras.layers.BatchNormalization())
        self.classification_model.add(keras.layers.Dense(self.number_of_classes))

        self.classification_model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=50, name="top50")
            ]
        )

    def evaluate(self, evaluation_dataset: tf.data.Dataset) -> Evaluation:
        assert self.classification_model is not None

        _, top_1_acc, top_5_acc, top_50_acc = self.classification_model.evaluate(evaluation_dataset)

        return Evaluation(
            top_1_accuracy=top_1_acc,
            top_5_accuracy=top_5_acc,
            top_50_accuracy=top_50_acc
        )


class SimilarityModel(ReidentModel):
    def compile(self) -> None:
        backbone_model = self.backbone_model()

        self.classification_model = keras.models.Sequential()
        self.classification_model.add(backbone_model)
        self.classification_model.add(keras.layers.Flatten())
        self.classification_model.add(keras.layers.Dense(512, use_bias=False))
        self.classification_model.add(keras.layers.Dense(self.number_of_classes))

        self.classification_model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=50, name="top50")
            ]
        )

    def evaluate(self, evaluation_dataset: tf.data.Dataset) -> Evaluation:
        assert self.classification_model is not None

        embedding_model = keras.Model(
            inputs=self.classification_model.inputs[0],
            outputs=self.classification_model.get_layer(index=-2).output
        )

        train_images, train_labels = [], []
        eval_images, eval_labels = [], []

        for images, labels in self.training_dataset:  # type: ignore
            train_images.append(images)
            train_labels.append(labels)

        for images, labels in self.validation_dataset:  # type: ignore
            train_images.append(images)
            train_labels.append(labels)

        for images, labels in evaluation_dataset:  # type: ignore
            eval_images.append(images)
            eval_labels.append(labels)

        train_images = tf.concat(train_images, axis=0)  # type: ignore
        train_labels = tf.concat(train_labels, axis=0).numpy()  # type: ignore

        eval_images = tf.concat(eval_images, axis=0)  # type: ignore
        eval_labels = tf.concat(eval_labels, axis=0).numpy()  # type: ignore

        train_embeddings = embedding_model.predict(train_images)
        eval_embeddings = embedding_model.predict(eval_images)

        train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        eval_embeddings_norm = eval_embeddings / np.linalg.norm(eval_embeddings, axis=1, keepdims=True)

        similarity_scores = np.dot(train_embeddings_norm, eval_embeddings_norm.transpose()).transpose()
        sorted_similarity_scores = np.argsort(similarity_scores, axis=1)[:, ::-1]

        top_1_indices = sorted_similarity_scores[:, :1]
        top_5_indices = sorted_similarity_scores[:, :5]
        top_50_indices = sorted_similarity_scores[:, :50]

        top_1_labels = train_labels[top_1_indices]
        top_5_labels = train_labels[top_5_indices]
        top_50_labels = train_labels[top_50_indices]

        top_1_acc = np.sum(top_1_labels[:, 0] == eval_labels) / len(eval_labels)
        top_5_acc = np.sum(np.any(top_5_labels == eval_labels[:, None], axis=1)) / len(eval_labels)
        top_50_acc = np.sum(np.any(top_50_labels == eval_labels[:, None], axis=1)) / len(eval_labels)

        return Evaluation(
            top_1_accuracy=top_1_acc,
            top_5_accuracy=top_5_acc,
            top_50_accuracy=top_50_acc
        )


def add_label_input(images: tf.Tensor, labels: tf.Tensor):
    return (
        {
            "images": images,
            "labels": labels
        },
        labels
    )


class ArcfaceTrainableModel(ReidentModel):
    def process_inputs(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        training_input, validation_input = super().process_inputs()

        training_input = training_input.map(add_label_input)
        validation_input = validation_input.map(add_label_input)

        return training_input, validation_input
    
    def compile(self) -> None:
        backbone_model = self.backbone_model()

        embedding_model = keras.models.Sequential()
        embedding_model.add(backbone_model)
        embedding_model.add(keras.layers.Flatten())
        embedding_model.add(keras.layers.Dense(512, use_bias=False))

        img_inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), name="images")
        label_inputs = keras.Input(shape=(), dtype=tf.int32, name="labels")

        embeddings = embedding_model(img_inputs)
        logits = ArcFaceLayer(self.number_of_classes)(embeddings, label_inputs)
        self.classification_model = keras.Model(inputs=[img_inputs, label_inputs], outputs=logits)

        self.classification_model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=50, name="top50")
            ]
        )


class SoftmaxArcfaceModel(ArcfaceTrainableModel):
    def evaluate(self, evaluation_dataset: tf.data.Dataset) -> Evaluation:
        assert self.classification_model is not None

        evaluation_input = evaluation_dataset.map(add_label_input)

        _, top_1_acc, top_5_acc, top_50_acc = self.classification_model.evaluate(evaluation_input)

        return Evaluation(
            top_1_accuracy=top_1_acc,
            top_5_accuracy=top_5_acc,
            top_50_accuracy=top_50_acc
        )

class SimilarityArcfaceModel(ArcfaceTrainableModel):
    def evaluate(self, evaluation_dataset: tf.data.Dataset) -> Evaluation:
        assert self.classification_model is not None

        embedding_model = self.classification_model.layers[1]

        train_images, train_labels = [], []
        eval_images, eval_labels = [], []

        for images, labels in self.training_dataset:  # type: ignore
            train_images.append(images)
            train_labels.append(labels)

        for images, labels in self.validation_dataset:  # type: ignore
            train_images.append(images)
            train_labels.append(labels)

        for images, labels in evaluation_dataset:  # type: ignore
            eval_images.append(images)
            eval_labels.append(labels)

        train_images = tf.concat(train_images, axis=0)  # type: ignore
        train_labels = tf.concat(train_labels, axis=0).numpy()  # type: ignore

        eval_images = tf.concat(eval_images, axis=0)  # type: ignore
        eval_labels = tf.concat(eval_labels, axis=0).numpy()  # type: ignore

        train_embeddings = embedding_model.predict(train_images)
        eval_embeddings = embedding_model.predict(eval_images)

        train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        eval_embeddings_norm = eval_embeddings / np.linalg.norm(eval_embeddings, axis=1, keepdims=True)

        similarity_scores = np.dot(train_embeddings_norm, eval_embeddings_norm.transpose()).transpose()
        sorted_similarity_scores = np.argsort(similarity_scores, axis=1)[:, ::-1]

        top_1_indices = sorted_similarity_scores[:, :1]
        top_5_indices = sorted_similarity_scores[:, :5]
        top_50_indices = sorted_similarity_scores[:, :50]

        top_1_labels = train_labels[top_1_indices]
        top_5_labels = train_labels[top_5_indices]
        top_50_labels = train_labels[top_50_indices]

        top_1_acc = np.sum(top_1_labels[:, 0] == eval_labels) / len(eval_labels)
        top_5_acc = np.sum(np.any(top_5_labels == eval_labels[:, None], axis=1)) / len(eval_labels)
        top_50_acc = np.sum(np.any(top_50_labels == eval_labels[:, None], axis=1)) / len(eval_labels)

        return Evaluation(
            top_1_accuracy=top_1_acc,
            top_5_accuracy=top_5_acc,
            top_50_accuracy=top_50_acc
        )


def get_reidentification_model(
    animal: str,
    loss_type: str,
    classification_type: str,
    *,
    batch_size: int = 32
) -> ReidentModel:
    out_dir = ROOT_DIR / ".out" / animal / loss_type / classification_type
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = ROOT_DIR / "dataset" / "out" / "images" / "train" / animal

    training_dataset = keras.utils.image_dataset_from_directory(
        directory=train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_DIMENSIONS,
        batch_size=batch_size
    )

    validation_dataset = keras.utils.image_dataset_from_directory(
        directory=train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_DIMENSIONS,
        batch_size=batch_size
    )

    if loss_type == "cross_entropy":
        if classification_type == "softmax":
            return SoftmaxModel(
                out_dir=out_dir,
                training_dataset=training_dataset,  # type: ignore
                validation_dataset=validation_dataset  # type: ignore
            )
        elif classification_type == "cosine_similarity":
            return SimilarityModel(
                out_dir=out_dir,
                training_dataset=training_dataset,  # type: ignore
                validation_dataset=validation_dataset  # type: ignore
            )
    elif loss_type == "arcface":
        if classification_type == "softmax":
            return SoftmaxArcfaceModel(
                out_dir=out_dir,
                training_dataset=training_dataset,  # type: ignore
                validation_dataset=validation_dataset  # type: ignore
            )
        elif classification_type == "cosine_similarity":
            return SimilarityArcfaceModel(
                out_dir=out_dir,
                training_dataset=training_dataset,  # type: ignore
                validation_dataset=validation_dataset  # type: ignore
            )
        
    raise NotImplementedError(f"{animal} - {loss_type} - {classification_type}")
