# Pet Face Identification Neural Networks Project

**Course**: EN 625.638 - Foundations of Neural Networks  
**Authors**: Matthew Tofano, McKenna Ahlm, and Bruce DeSimas  
**Dataset**: PetFace Dataset

## Project Overview

This project implements and compares multiple deep learning approaches for **individual animal identification** using facial features. Unlike species classification, our system identifies specific individual animals across 13 different pet species, demonstrating advanced neural network techniques including transfer learning, metric learning, and custom loss functions.

## Research Question

**How do different loss functions and classification methods affect performance in animal face re-identification tasks?**

We systematically compare:
- **Loss Functions**: Traditional cross-entropy vs ArcFace angular margin loss
- **Classification Methods**: Direct softmax classification vs cosine similarity matching
- **Model Performance**: Across 13 different animal species with varying facial characteristics

## Key Technical Contributions

### 1. Custom ArcFace Implementation
- Angular margin loss layer implemented from scratch in TensorFlow/Keras
- Improves feature discrimination for face identification tasks
- Normalizes embeddings in angular space for better class separation

### 2. Comprehensive Model Comparison
- **4 model configurations** systematically evaluated per animal species
- **52 total experiments** (4 models × 13 animals) 
- Controlled comparison of training vs evaluation strategies

### 3. Transfer Learning with ResNet50
- Pre-trained ImageNet backbone with strategic fine-tuning
- Last 30 layers trainable, earlier layers frozen
- Optimized for animal facial feature extraction

## Model Architecture Overview

All models share a common backbone with different training and evaluation strategies:

```
Input (224×224×3) → ResNet50 Backbone → Feature Extraction → Classification/Embedding
                        ↓
              [Transfer Learning Strategy]
                        ↓
            Cross-Entropy Loss    vs    ArcFace Angular Loss
                        ↓
            Softmax Classification  vs  Cosine Similarity Matching
```

### Four Model Configurations:
1. **Cross-Entropy + Softmax**: Traditional classification baseline
2. **Cross-Entropy + Cosine Similarity**: Embedding-based matching with standard training
3. **ArcFace + Softmax**: Advanced training with traditional evaluation
4. **ArcFace + Cosine Similarity**: State-of-the-art approach combining both advances

📖 **For detailed technical documentation, see [Face Identification Models](petface/face_identification/README.md)**

## Dataset Details

**Target Species (13)**: cat, chimp, chinchilla, degus, dog, ferret, guineapig, hamster, hedgehog, javasparrow, parakeet, pig, rabbit

**Image Specifications**:
- Input size: 224×224×3 RGB  
- Organized by individual animal ID within species folders
- Training/validation split: 80/20 with stratification

**Evaluation Strategy**:
- **Top-k Accuracy Metrics**: Top-1, Top-5, and Top-50 accuracy
- **Cross-validation**: Systematic comparison across all 13 species
- **Controlled Experiments**: Same data splits for fair model comparison

## Project Structure

```
petface/
├── face_identification/
│   ├── model.py                  # Main neural network implementations
│   └── README.md                 # Detailed technical documentation
├── const.py                      # Configuration constants
scripts/
├── run_training.py              # Automated training pipeline
├── run_training_and_evaluation.py # Training + evaluation
└── process_train_test_splits.py # Dataset preparation
reports/                         # Auto-generated evaluation notebooks
```

## Running Experiments

### Full Training Pipeline
```bash
python scripts/run_training.py
# Trains all 4 model combinations for each of the 13 animals (52 total experiments)
```

### Training + Evaluation
```bash
python scripts/run_training_and_evaluation.py  
# Runs complete experiments and generates evaluation reports
```

## Educational Value

This project demonstrates key neural network concepts:

- **Transfer Learning**: Strategic fine-tuning of pre-trained models
- **Metric Learning**: Custom loss functions for improved embeddings  
- **Model Architecture Design**: Systematic comparison of training vs evaluation strategies
- **Experimental Methodology**: Controlled comparison across multiple variables
- **Real-world Applications**: Face identification systems and similarity matching

## Results

The project generates **52 individual experiments** (4 models × 13 animals) with results automatically documented in Jupyter notebooks within the `reports/` directory. This systematic evaluation enables comparison of:

- Cross-entropy vs ArcFace loss effectiveness
- Classification vs similarity-based identification performance  
- Model performance variation across different animal species

## Setup Instructions

For environment setup, dataset download, and installation instructions, see [SETUP.md](SETUP.md).