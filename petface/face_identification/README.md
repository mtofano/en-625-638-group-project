# Face Identification Models

This module implements various neural network architectures for pet face identification and re-identification. The system compares different loss functions and classification approaches to identify individual animals across multiple species.

## Architecture Overview

All models use **transfer learning** with ResNet50 as the backbone, pre-trained on ImageNet, then fine-tuned for animal face identification. The project systematically compares:

- **Loss Functions**: Cross-entropy vs ArcFace angular margin loss
- **Classification Methods**: Softmax classification vs cosine similarity matching

## Model Classes

### 1. Base Classes

#### `ReidentModel` (Abstract Base Class)
- Provides common functionality for all identification models
- Handles model checkpointing, training history, and data pipeline optimization
- Uses ResNet50 backbone with transfer learning (last 30 layers fine-tuned)
- Implements callbacks for early stopping and best model saving

#### `ArcFaceLayer` (Custom Keras Layer)
- Implements angular margin loss for improved feature discrimination
- Normalizes embeddings and weights to unit vectors for angular distance computation
- Adds angular margin only to ground truth classes during training
- **Parameters**: 
  - `margin`: Angular margin in radians (default: 0.05)
  - `scale`: Scaling factor for numerical stability (default: 20)

### 2. Cross-Entropy Models

#### `SoftmaxModel`
**Traditional baseline classification approach**

- **Architecture**: ResNet50 → Flatten → Dense(512) → BatchNorm → Dense(num_classes)
- **Training**: Standard cross-entropy loss with Adam optimizer
- **Evaluation**: Direct softmax probability classification
- **Use Case**: Traditional multi-class classification baseline

#### `SimilarityModel` 
**Embedding-based identification with cosine similarity**

- **Architecture**: ResNet50 → Flatten → Dense(512) → Dense(num_classes)
- **Training**: Cross-entropy loss (same as SoftmaxModel)
- **Evaluation**: Extract embeddings, compute cosine similarity, rank by similarity
- **Use Case**: Simulates real-world identification systems with gallery matching

### 3. ArcFace Models

#### `SoftmaxArcfaceModel`
**ArcFace training with traditional evaluation**

- **Architecture**: ResNet50 → Flatten → Dense(512) → ArcFaceLayer
- **Training**: ArcFace angular margin loss for better embeddings
- **Evaluation**: Standard softmax classification using trained model
- **Use Case**: Combines improved ArcFace training with simple evaluation

#### `SimilarityArcfaceModel`
**Most sophisticated approach - ArcFace training + similarity evaluation**

- **Architecture**: ResNet50 → Flatten → Dense(512) → ArcFaceLayer  
- **Training**: ArcFace angular margin loss
- **Evaluation**: Extract ArcFace embeddings, cosine similarity matching
- **Use Case**: State-of-the-art approach mimicking production face recognition systems

## Technical Implementation Details

### Transfer Learning Strategy
```python
# Load pre-trained ResNet50 without classification head
backbone = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    pooling="avg"
)

# Freeze initial layers, fine-tune last 30 layers
for layer in backbone.layers[:-30]:
    layer.trainable = False
for layer in backbone.layers[-30:]:
    layer.trainable = True
```

### ArcFace Loss Implementation
```python
# Normalize embeddings and weights to unit vectors
embeddings = tf.nn.l2_normalize(inputs, axis=1)
weights = tf.nn.l2_normalize(self.W, axis=0)

# Compute angular distance and add margin
theta = tf.acos(tf.clip_by_value(logits, -1+1e-7, 1-1e-7))
cos_theta_m = tf.cos(theta + self._margin)

# Apply margin only to ground truth class
one_hot = tf.one_hot(labels, depth=self._num_classes)
final_logits = logits * (1 - one_hot) + cos_theta_m * one_hot
return final_logits * self._scale
```

### Similarity-Based Evaluation
```python
# Extract normalized embeddings
train_embeddings = embedding_model.predict(gallery_images)
eval_embeddings = embedding_model.predict(query_images)

train_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
eval_norm = eval_embeddings / np.linalg.norm(eval_embeddings, axis=1, keepdims=True)

# Compute cosine similarity and rank
similarity_scores = np.dot(train_norm, eval_norm.T).T
ranked_indices = np.argsort(similarity_scores, axis=1)[:, ::-1]
```

## Training Configuration

- **Optimizer**: Adam
- **Epochs**: 25 with early stopping (patience=5 on validation loss)
- **Batch Size**: 32
- **Image Size**: 224×224×3
- **Data Split**: 80% training, 20% validation
- **Callbacks**: ModelCheckpoint (save best), EarlyStopping

## Evaluation Metrics

All models report **top-k accuracy** metrics:

- **Top-1 Accuracy**: Exact match - model's first prediction is correct
- **Top-5 Accuracy**: Correct answer appears in top 5 predictions  
- **Top-50 Accuracy**: Correct answer appears in top 50 predictions

These metrics assess both precision (top-1) and the model's ability to narrow down possibilities (top-5, top-50).

## Model Factory Function

```python
def get_reidentification_model(animal, loss_type, classification_type, batch_size=32):
    """
    Create model instance for specified configuration.
    
    Args:
        animal: Species name ('cat', 'dog', 'rabbit', etc.)
        loss_type: 'cross_entropy' or 'arcface'  
        classification_type: 'softmax' or 'cosine_similarity'
        
    Returns:
        Configured ReidentModel instance
    """
```

## Experimental Design

The project systematically evaluates **4 model configurations** per animal species:

1. **cross_entropy + softmax**: Traditional classification baseline
2. **cross_entropy + cosine_similarity**: Standard embeddings with similarity matching
3. **arcface + softmax**: Improved training with traditional evaluation
4. **arcface + cosine_similarity**: State-of-the-art approach combining both advances

## File Organization

```
petface/face_identification/
├── model.py           # Main model implementations
├── README.md          # This documentation
└── __init__.py        # Module initialization
```

## Usage Example

```python
from petface.face_identification.model import get_reidentification_model

# Create and train a model
model = get_reidentification_model(
    animal="cat",
    loss_type="arcface", 
    classification_type="cosine_similarity"
)

# Train the model
model.train()

# Evaluate on test data
results = model.evaluate(test_dataset)
print(f"Top-1 Accuracy: {results['top_1_accuracy']:.3f}")
```

## Neural Network Concepts Demonstrated

1. **Transfer Learning**: Leveraging ImageNet features for specialized tasks
2. **Metric Learning**: ArcFace loss for discriminative embedding learning
3. **Angular Margin Loss**: Geometric approach to improve class separation
4. **Embedding Normalization**: L2 normalization for cosine similarity computation
5. **Multi-task Evaluation**: Classification vs similarity-based identification
6. **Regularization**: BatchNormalization, early stopping, model checkpointing
7. **Data Pipeline Optimization**: Prefetching for training efficiency

This implementation serves as a comprehensive example of modern deep learning techniques applied to animal identification, suitable for educational purposes in neural networks courses.