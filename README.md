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
