"""
Pet Face Identification Neural Network Models

This module implements various neural network architectures for pet face identification/re-identification.
The project compares different loss functions (cross-entropy vs ArcFace) and classification methods 
(softmax vs cosine similarity) for identifying individual animals across multiple species.

Key Components:
- ArcFaceLayer: Custom layer implementing ArcFace loss for better feature learning
- ReidentModel: Abstract base class for all identification models
- SoftmaxModel: Traditional softmax classification approach
- SimilarityModel: Cosine similarity-based identification
- ArcfaceTrainableModel: Base class for models using ArcFace loss
- SoftmaxArcfaceModel: ArcFace loss with softmax classification
- SimilarityArcfaceModel: ArcFace loss with similarity-based identification
"""

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

# Image dimensions used throughout the project
IMG_DIMENSIONS = (IMG_HEIGHT, IMG_WIDTH)


class ArcFaceLayer(keras.layers.Layer):
    """
    Custom ArcFace (Angular Margin Loss) layer implementation.
    
    ArcFace is an advanced loss function that improves upon traditional softmax by adding 
    angular margin in the angular space, leading to more discriminative feature embeddings.
    This is particularly useful for face identification tasks where we need to distinguish
    between many different individuals.
    
    The layer works by:
    1. Normalizing both input embeddings and learned weights to unit vectors
    2. Computing angular distance between embeddings and class weights
    3. Adding angular margin only to the ground truth class
    4. Scaling the final logits for numerical stability
    
    Args:
        num_classes: Number of individual animals/identities to classify
        margin: Angular margin in radians (default 0.05). Larger values create wider gaps between classes
        scale: Scaling factor for logits (default 20). Controls the sharpness of the distribution
    """
    def __init__(
        self, 
        num_classes: int, 
        *, 
        margin: float = 0.05, 
        scale: int = 20, 
        **kwargs
    ):
        self._num_classes = num_classes
        self._margin = margin  # Angular margin to separate classes
        self._scale = scale    # Logit scaling factor
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Initialize learnable weight matrix for each class."""
        embedding_dim = input_shape[-1]

        # Create weight matrix: each column represents a class center in embedding space
        self.W = self.add_weight(
            name="W",
            shape=(embedding_dim, self._num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )
    
    def call(self, inputs: tf.Tensor, labels: tf.Tensor):
        """
        Forward pass implementing ArcFace loss computation.
        
        Args:
            inputs: Feature embeddings from the network [batch_size, embedding_dim]
            labels: Ground truth class labels [batch_size]
            
        Returns:
            Scaled logits with angular margin applied to correct classes
        """
        # Normalize embeddings and weights to unit vectors (important for angular distance)
        embeddings = tf.nn.l2_normalize(inputs, axis=1)
        weights = tf.nn.l2_normalize(self.W, axis=0)

        # Compute cosine similarity between embeddings and class weights
        logits = tf.matmul(embeddings, weights)

        # Convert cosine similarity to angle (theta)
        theta = tf.acos(tf.clip_by_value(logits, -1 + 1e-7, 1 - 1e-7))

        # Add angular margin to the ground truth class angle
        cos_theta_m = tf.cos(theta + self._margin)

        # Create one-hot encoding for ground truth labels
        one_hot = tf.one_hot(labels, depth=self._num_classes)

        # Apply margin only to the correct class, keep original logits for other classes
        final_logits = logits * (1 - one_hot) + cos_theta_m * one_hot

        # Scale logits for numerical stability and convergence
        return final_logits * self._scale
    

class Evaluation(TypedDict):
    """
    Structured evaluation results for model performance assessment.
    
    For animal face identification, we use top-k accuracy metrics because:
    - top_1_accuracy: Exact match - the model's first guess is correct
    - top_5_accuracy: Model's confidence - correct answer in top 5 predictions
    - top_50_accuracy: Model's understanding - correct answer in top 50 predictions
    
    These metrics help assess how well the model distinguishes between individual animals.
    """
    top_1_accuracy: float
    top_5_accuracy: float
    top_50_accuracy: float


class ReidentModel(ABC):
    """
    Abstract base class for all animal face re-identification models.
    
    This class provides the common functionality needed for training and evaluating
    different neural network architectures for pet face identification. The project
    compares multiple approaches:
    1. Loss functions: Traditional cross-entropy vs ArcFace angular margin loss
    2. Classification methods: Softmax classification vs cosine similarity matching
    
    All models use transfer learning with ResNet50 as the backbone, pre-trained on ImageNet.
    """
    def __init__(
        self,
        out_dir: pathlib.Path,
        training_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset
    ) -> None:
        self.out_dir = out_dir  # Directory for saving model outputs and checkpoints
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.classification_model: keras.Model | None = None  # The compiled model

    @property
    def number_of_classes(self) -> int:
        """Number of individual animals in the dataset (each animal is a class)."""
        return len(self.training_dataset.class_names)  # type: ignore
    
    @property
    def checkpoint_file(self) -> pathlib.Path:
        """Path to save/load model weights during training."""
        checkpoint_file = self.out_dir / "checkpoints" / "cp.weights.h5"
        if not checkpoint_file.parent.exists():
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        return checkpoint_file

    @property
    def history_file(self) -> pathlib.Path:
        """Path to save training history (loss, accuracy over epochs)."""
        history_file = self.out_dir / "history" / "history.json"
        if not history_file.parent.exists():
            history_file.parent.mkdir(parents=True, exist_ok=True)
        return history_file 
    
    @abstractmethod
    def compile(self) -> None:
        """Build and compile the model architecture. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def evaluate(self, evaluation_dataset: tf.data.Dataset) -> Evaluation:
        """Evaluate model performance on test data. Must be implemented by subclasses."""
        ... 
    
    def backbone_model(self) -> keras.Model:
        """
        Create the backbone feature extractor using transfer learning.
        
        Uses ResNet50 pre-trained on ImageNet as the foundation. This leverages
        learned visual features from millions of images, then fine-tunes for our
        specific task of animal face identification.
        
        Transfer learning strategy:
        1. Load ResNet50 without the final classification layer (include_top=False)
        2. Freeze most layers to preserve learned features
        3. Unfreeze the last 30 layers for fine-tuning on our animal data
        """
        backbone = keras.applications.ResNet50(
            include_top=False,  # Remove final classification layer
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH),
            pooling="avg",  # Global average pooling instead of flattening
            classes=self.number_of_classes,
            weights="imagenet",  # Use ImageNet pre-trained weights
        )

        # Initially freeze all layers to preserve pre-trained features
        for layer in backbone.layers:
            layer.trainable = False

        # Unfreeze the last 30 layers for fine-tuning on our specific data
        # This allows the model to adapt high-level features for animal faces
        for layer in backbone.layers[-30:]:
            layer.trainable = True

        return backbone

    def process_inputs(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Optimize data pipeline for training efficiency.
        
        Prefetching allows the data pipeline to prepare the next batch while
        the current batch is being processed, reducing training time.
        """
        training_dataset = self.training_dataset.prefetch(tf.data.AUTOTUNE)  # type: ignore
        validation_dataset = self.validation_dataset.prefetch(tf.data.AUTOTUNE)  # type: ignore

        return training_dataset, validation_dataset

    def train(self) -> None:
        """
        Train the model with callbacks for monitoring and early stopping.
        
        Training strategy:
        1. Save best model weights during training
        2. Stop early if validation loss stops improving (prevent overfitting)
        3. Train for up to 25 epochs
        4. Save training history for analysis
        """
        if not self.classification_model:
            self.compile()

        assert self.classification_model is not None

        print(self.classification_model.summary())

        # Save the best model weights during training
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_file,
            save_weights_only=True,
            save_best_only=True,
            verbose=1
        )

        # Stop training early if validation loss doesn't improve (prevents overfitting)
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Wait 5 epochs before stopping
            restore_best_weights=True
        )

        training_input, validation_input = self.process_inputs()

        # Train the model
        history = self.classification_model.fit(
            training_input,
            validation_data=validation_input,
            epochs=25,
            callbacks=[cp_callback, early_stop]
        )

        # Save training history for later analysis and plotting
        with self.history_file.open("w") as f:
            json.dump(history.history, f)

    def load_from_checkpoint(self) -> None:
        """Load previously trained model weights from checkpoint file."""
        if not self.classification_model:
            self.compile()

        assert self.classification_model is not None

        self.classification_model.load_weights(self.checkpoint_file)

    def display_training_history(self) -> None:
        """
        Create interactive plots showing training progress over time.
        
        Visualizes key metrics to assess model performance:
        - Accuracy metrics: How well the model identifies animals correctly
        - Loss values: How confident the model is in its predictions
        - Training vs Validation: Helps identify overfitting
        
        The plots help determine if the model is learning effectively and
        whether it generalizes well to unseen data.
        """
        with self.history_file.open("r") as f:
            history = json.load(f)

        epochs = list(range(1, len(history["sparse_categorical_accuracy"])))

        # Create subplot with dual y-axes (accuracy and loss)
        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )

        # Plot training accuracy metrics
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
        
        # Plot validation accuracy metrics
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
        
        # Plot loss metrics on secondary y-axis
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
    """
    Traditional classification model using cross-entropy loss and softmax activation.
    
    This represents the baseline approach for animal face identification:
    1. Extract features using ResNet50 backbone
    2. Add dense layers for classification
    3. Use standard cross-entropy loss
    4. Make predictions using softmax probabilities
    
    This approach treats each individual animal as a separate class and
    learns to classify images directly into these classes.
    """
    def compile(self) -> None:
        """
        Build the traditional classification model architecture.
        
        Architecture:
        1. ResNet50 backbone (feature extraction)
        2. Flatten layer (convert 2D features to 1D)
        3. Dense layer (512 units) + BatchNormalization (regularization)
        4. Output layer (number of animal classes)
        
        Uses Adam optimizer and cross-entropy loss for multi-class classification.
        """
        backbone_model = self.backbone_model()

        # Build the classification head
        self.classification_model = keras.models.Sequential()
        self.classification_model.add(backbone_model)
        self.classification_model.add(keras.layers.Flatten())
        self.classification_model.add(keras.layers.Dense(512, use_bias=False))
        self.classification_model.add(keras.layers.BatchNormalization())  # Helps with training stability
        self.classification_model.add(keras.layers.Dense(self.number_of_classes))  # Output layer

        # Compile with standard multi-class classification setup
        self.classification_model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),  # Top-1 accuracy
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),   # Top-5 accuracy
                keras.metrics.SparseTopKCategoricalAccuracy(k=50, name="top50")  # Top-50 accuracy
            ]
        )

    def evaluate(self, evaluation_dataset: tf.data.Dataset) -> Evaluation:
        """
        Standard evaluation using softmax probabilities.
        
        The model outputs class probabilities and we measure how often
        the correct animal appears in the top-k predictions.
        """
        assert self.classification_model is not None

        # Get evaluation metrics directly from the compiled model
        _, top_1_acc, top_5_acc, top_50_acc = self.classification_model.evaluate(evaluation_dataset)

        return Evaluation(
            top_1_accuracy=top_1_acc,
            top_5_accuracy=top_5_acc,
            top_50_accuracy=top_50_acc
        )


class SimilarityModel(ReidentModel):
    """
    Cosine similarity-based identification model.
    
    This approach learns feature embeddings and uses cosine similarity for identification:
    1. Extract features using ResNet50 backbone
    2. Learn 512-dimensional embeddings
    3. At evaluation, compare query embeddings to all known embeddings
    4. Rank by cosine similarity to find matches
    
    This mimics face recognition systems where you compare a face to a database
    of known faces to find the best match.
    """
    def compile(self) -> None:
        """
        Build the similarity-based model architecture.
        
        Note: During training, this model is trained exactly like the softmax model
        using cross-entropy loss. The difference is in evaluation - instead of using
        softmax probabilities, we extract embeddings and use cosine similarity.
        """
        backbone_model = self.backbone_model()

        # Build model for learning embeddings
        self.classification_model = keras.models.Sequential()
        self.classification_model.add(backbone_model)
        self.classification_model.add(keras.layers.Flatten())
        self.classification_model.add(keras.layers.Dense(512, use_bias=False))  # Embedding layer
        self.classification_model.add(keras.layers.Dense(self.number_of_classes))  # Classification layer

        # Train using standard classification loss (same as SoftmaxModel)
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
        """
        Evaluate using cosine similarity instead of softmax probabilities.
        
        Process:
        1. Extract embeddings for all training + validation images (reference gallery)
        2. Extract embeddings for evaluation images (queries)
        3. Compute cosine similarity between query and gallery embeddings
        4. Rank gallery images by similarity and check if correct animal is in top-k
        
        This simulates a real-world identification system where you compare
        a new image against a database of known animals.
        """
        assert self.classification_model is not None

        # Create embedding model (remove the final classification layer)
        embedding_model = keras.Model(
            inputs=self.classification_model.inputs[0],
            outputs=self.classification_model.get_layer(index=-2).output  # Get embeddings before final layer
        )

        # Collect all images and labels from training and validation (gallery)
        train_images, train_labels = [], []
        eval_images, eval_labels = [], []

        # Training set becomes part of our reference gallery
        for images, labels in self.training_dataset:  # type: ignore
            train_images.append(images)
            train_labels.append(labels)

        # Validation set also becomes part of our reference gallery
        for images, labels in self.validation_dataset:  # type: ignore
            train_images.append(images)
            train_labels.append(labels)

        # Evaluation set contains our query images
        for images, labels in evaluation_dataset:  # type: ignore
            eval_images.append(images)
            eval_labels.append(labels)

        # Concatenate all batches
        train_images = tf.concat(train_images, axis=0)  # type: ignore
        train_labels = tf.concat(train_labels, axis=0).numpy()  # type: ignore

        eval_images = tf.concat(eval_images, axis=0)  # type: ignore
        eval_labels = tf.concat(eval_labels, axis=0).numpy()  # type: ignore

        # Extract embeddings for gallery and queries
        train_embeddings = embedding_model.predict(train_images)
        eval_embeddings = embedding_model.predict(eval_images)

        # Normalize embeddings for cosine similarity computation
        train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        eval_embeddings_norm = eval_embeddings / np.linalg.norm(eval_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity between all queries and gallery images
        similarity_scores = np.dot(train_embeddings_norm, eval_embeddings_norm.transpose()).transpose()
        
        # Sort by similarity (descending order)
        sorted_similarity_scores = np.argsort(similarity_scores, axis=1)[:, ::-1]

        # Get top-k most similar images
        top_1_indices = sorted_similarity_scores[:, :1]
        top_5_indices = sorted_similarity_scores[:, :5]
        top_50_indices = sorted_similarity_scores[:, :50]

        # Get labels of top-k similar images
        top_1_labels = train_labels[top_1_indices]
        top_5_labels = train_labels[top_5_indices]
        top_50_labels = train_labels[top_50_indices]

        # Calculate accuracy: does the correct label appear in top-k?
        top_1_acc = np.sum(top_1_labels[:, 0] == eval_labels) / len(eval_labels)
        top_5_acc = np.sum(np.any(top_5_labels == eval_labels[:, None], axis=1)) / len(eval_labels)
        top_50_acc = np.sum(np.any(top_50_labels == eval_labels[:, None], axis=1)) / len(eval_labels)

        return Evaluation(
            top_1_accuracy=top_1_acc,
            top_5_accuracy=top_5_acc,
            top_50_accuracy=top_50_acc
        )


def add_label_input(images: tf.Tensor, labels: tf.Tensor):
    """
    Helper function to format data for ArcFace models.
    
    ArcFace layer needs both images and labels during training to apply
    the angular margin. This function restructures the data into the
    required format with named inputs.
    
    Args:
        images: Batch of input images
        labels: Corresponding ground truth labels
    
    Returns:
        Tuple of (dict with named inputs, labels) suitable for ArcFace training
    """
    return (
        {
            "images": images,
            "labels": labels
        },
        labels
    )


class ArcfaceTrainableModel(ReidentModel):
    """
    Base class for models using ArcFace angular margin loss.
    
    ArcFace improves upon traditional cross-entropy by adding angular margin
    in the embedding space. This creates larger gaps between different classes
    and leads to more discriminative features for face identification.
    
    The key difference from traditional models is that ArcFace requires
    both images and labels during training to compute the angular margin.
    """
    def process_inputs(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare data for ArcFace training by restructuring inputs.
        
        ArcFace layer needs labels during forward pass to apply angular margin,
        so we restructure the data to provide both images and labels as inputs.
        """
        training_input, validation_input = super().process_inputs()

        # Restructure data to provide both images and labels as model inputs
        training_input = training_input.map(add_label_input)
        validation_input = validation_input.map(add_label_input)

        return training_input, validation_input
    
    def compile(self) -> None:
        """
        Build model architecture with ArcFace layer.
        
        Architecture:
        1. ResNet50 backbone (feature extraction)
        2. Flatten + Dense layer (embedding generation)
        3. ArcFace layer (applies angular margin during training)
        
        The model has two inputs (images and labels) because ArcFace
        needs the ground truth labels to apply angular margin.
        """
        backbone_model = self.backbone_model()

        # Build embedding extraction network
        embedding_model = keras.models.Sequential()
        embedding_model.add(backbone_model)
        embedding_model.add(keras.layers.Flatten())
        embedding_model.add(keras.layers.Dense(512, use_bias=False))  # Generate 512-dim embeddings

        # Define dual inputs: images and labels
        img_inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), name="images")
        label_inputs = keras.Input(shape=(), dtype=tf.int32, name="labels")

        # Extract embeddings and apply ArcFace layer
        embeddings = embedding_model(img_inputs)
        logits = ArcFaceLayer(self.number_of_classes)(embeddings, label_inputs)
        
        # Create model with dual inputs and ArcFace outputs
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
    """
    ArcFace loss with traditional softmax evaluation.
    
    This model trains with ArcFace angular margin loss but evaluates
    using standard softmax probabilities. This combines the benefits
    of ArcFace training (better feature learning) with simple evaluation.
    """
    def evaluate(self, evaluation_dataset: tf.data.Dataset) -> Evaluation:
        """
        Evaluate using the trained ArcFace model with softmax probabilities.
        
        Even though we trained with ArcFace loss, we can still use the model
        to generate class probabilities for standard classification evaluation.
        """
        assert self.classification_model is not None

        # Restructure evaluation data to match ArcFace input format
        evaluation_input = evaluation_dataset.map(add_label_input)

        # Standard evaluation using the compiled metrics
        _, top_1_acc, top_5_acc, top_50_acc = self.classification_model.evaluate(evaluation_input)

        return Evaluation(
            top_1_accuracy=top_1_acc,
            top_5_accuracy=top_5_acc,
            top_50_accuracy=top_50_acc
        )

class SimilarityArcfaceModel(ArcfaceTrainableModel):
    """
    ArcFace loss with cosine similarity evaluation.
    
    This model combines ArcFace training (for better embeddings) with
    cosine similarity evaluation (for more realistic identification scenarios).
    This represents the most sophisticated approach in our comparison.
    """
    def evaluate(self, evaluation_dataset: tf.data.Dataset) -> Evaluation:
        """
        Evaluate using ArcFace-trained embeddings with cosine similarity.
        
        This combines the best of both worlds:
        1. ArcFace training produces better discriminative embeddings
        2. Cosine similarity evaluation matches real-world identification systems
        
        The process is similar to SimilarityModel evaluation but uses embeddings
        learned with ArcFace angular margin loss.
        """
        assert self.classification_model is not None

        # Extract the embedding model from the ArcFace architecture
        # Layer 1 is the embedding model (before the ArcFace layer)
        embedding_model = self.classification_model.layers[1]

        # Collect gallery images (training + validation)
        train_images, train_labels = [], []
        eval_images, eval_labels = [], []

        for images, labels in self.training_dataset:  # type: ignore
            train_images.append(images)
            train_labels.append(labels)

        for images, labels in self.validation_dataset:  # type: ignore
            train_images.append(images)
            train_labels.append(labels)

        # Collect query images (evaluation set)
        for images, labels in evaluation_dataset:  # type: ignore
            eval_images.append(images)
            eval_labels.append(labels)

        # Concatenate all batches
        train_images = tf.concat(train_images, axis=0)  # type: ignore
        train_labels = tf.concat(train_labels, axis=0).numpy()  # type: ignore

        eval_images = tf.concat(eval_images, axis=0)  # type: ignore
        eval_labels = tf.concat(eval_labels, axis=0).numpy()  # type: ignore

        # Extract ArcFace-trained embeddings
        train_embeddings = embedding_model.predict(train_images)
        eval_embeddings = embedding_model.predict(eval_images)

        # Normalize for cosine similarity
        train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        eval_embeddings_norm = eval_embeddings / np.linalg.norm(eval_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity and rank results
        similarity_scores = np.dot(train_embeddings_norm, eval_embeddings_norm.transpose()).transpose()
        sorted_similarity_scores = np.argsort(similarity_scores, axis=1)[:, ::-1]

        # Extract top-k matches
        top_1_indices = sorted_similarity_scores[:, :1]
        top_5_indices = sorted_similarity_scores[:, :5]
        top_50_indices = sorted_similarity_scores[:, :50]

        top_1_labels = train_labels[top_1_indices]
        top_5_labels = train_labels[top_5_indices]
        top_50_labels = train_labels[top_50_indices]

        # Calculate top-k accuracies
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
    """
    Factory function to create the appropriate model for experimentation.
    
    This project compares 4 different model configurations:
    1. cross_entropy + softmax: Traditional classification (baseline)
    2. cross_entropy + cosine_similarity: Standard embeddings with similarity matching
    3. arcface + softmax: ArcFace training with traditional evaluation
    4. arcface + cosine_similarity: ArcFace training with similarity evaluation (most advanced)
    
    Args:
        animal: Animal species to train on (e.g., 'dog', 'cat', 'rabbit')
        loss_type: 'cross_entropy' or 'arcface'
        classification_type: 'softmax' or 'cosine_similarity'
        batch_size: Training batch size
    
    Returns:
        Configured model ready for training and evaluation
    """
    # Create output directory for this specific configuration
    out_dir = ROOT_DIR / ".out" / animal / loss_type / classification_type
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset for the specified animal
    train_dir = ROOT_DIR / "dataset" / "out" / "images" / "train" / animal

    # Create training dataset (80% of data)
    training_dataset = keras.utils.image_dataset_from_directory(
        directory=train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,  # Fixed seed for reproducibility
        image_size=IMG_DIMENSIONS,
        batch_size=batch_size
    )

    # Create validation dataset (20% of data)
    validation_dataset = keras.utils.image_dataset_from_directory(
        directory=train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,  # Same seed ensures consistent splits
        image_size=IMG_DIMENSIONS,
        batch_size=batch_size
    )

    # Return the appropriate model based on configuration
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
