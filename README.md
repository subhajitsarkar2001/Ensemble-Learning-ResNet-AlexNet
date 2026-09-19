##  **Rice Leaf Disease Detection using Deep Learning(Phase 2)**

##  **Overview**

This repository contains **Phase 2** of a deep learning based rice leaf disease detection project.

The objective of this phase is to improve the reliability of rice leaf classification by combining predictions from two different deep learning models:

- **AlexNet**
- **ResNet-18**
- Weighted ensemble learning applied  on both

Instead of relying on a single model, the system extracts class probabilities from both networks and combines them using a **weighted probability ensemble**.

The implementation brings together **TensorFlow/Keras** for AlexNet inference and **PyTorch** for ResNet-18 inference, followed by probability-level fusion using Python and scikit-learn.

## Dataset

The same rice leaf validation dataset used during model development is used to evaluate both individual models and the final ensemble.

The dataset contains **8 rice leaf classes**:

- Bacterial Leaf Blight
- Brown Spot
- Healthy Rice Leaf
- Leaf Blast
- Leaf scald
- Narrow Brown Leaf Spot
- Rice Hispa
- Sheath Blight

### Validation Dataset


Validation Images: 186

Number of Classes: 8

## **Dataset Structure**

Rice leaf disease Dataset




└── Validation data/
    
    ├── Bacterial Leaf Blight/
    ├── Brown Spot/
    ├── Healthy Rice Leaf/
    ├── Leaf Blast/
    ├── Leaf scald/
    ├── Narrow Brown Leaf Spot/
    ├── Rice Hispa/
    └── Sheath Blight/

The class ordering is kept consistent while generating predictions from both models so that their probability vectors correspond to the same disease classes.

## **Models Used**

The ensemble consists of two independently trained deep learning models.

**1. AlexNet**

The AlexNet model developed during Phase 1 is loaded from:

      rice_leaf_alexnet_model.keras

AlexNet processes the validation images at:

      227 × 227

The model generates a probability distribution over the eight disease classes.

**2. ResNet-18**

A trained ResNet-18 model is loaded using PyTorch.

The validation images are resized to:

     224 × 224

and normalized using the standard ImageNet mean and standard deviation:

     Mean = [0.485, 0.456, 0.406]

     Std = [0.229, 0.224, 0.225]

The ResNet-18 model produces class probabilities through the Softmax function.

## **Ensemble Method**

The two models do not contribute equally to the final prediction.

A weighted probability fusion strategy is used.

**Assigned Weights**

     ResNet-18  → 60%

     AlexNet    → 40%

The final ensemble probability for each class is calculated as:

     Ensemble Probability
      =
     (0.60 × ResNet-18 Probability)
     +
     (0.40 × AlexNet Probability)

The class having the highest combined probability becomes the final ensemble prediction.

## **Ensemble Workflow**

                 Validation Image
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
           AlexNet             ResNet-18
              │                   │
              ▼                   ▼
       Class Probabilities   Class Probabilities
              │                   │
              │                   │
              └─────────┬─────────┘
                        ▼
              Weighted Probability
                     Fusion
                        │
                        ▼
                Final Prediction
                        │
                        ▼
              Rice Leaf Disease Class

## **Implementation**

The ensemble pipeline performs the following operations:

1)Load the validation dataset.

2)Load the trained AlexNet model.

3)Generate AlexNet class probabilities.

4)Calculate AlexNet validation accuracy.

5)Release TensorFlow model resources.

6)Load the trained ResNet-18 model.

7)Apply ResNet-18 preprocessing.

8)Generate ResNet-18 class probabilities.

9)Calculate ResNet-18 validation accuracy.

10)Combine both probability distributions using weighted averaging.

11)Select the class with the highest combined probability.

12)Evaluate the final ensemble using multiple classification metrics.
