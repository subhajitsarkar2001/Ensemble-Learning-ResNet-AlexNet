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

**Model Accuracy Comparison**

The validation results obtained from the three prediction approaches are:

Model	          Validation Accuracy
AlexNet	            69.35%
ResNet-18	        97.31%
Weighted Ensemble	96.24%

The results demonstrate that the ResNet-18 model provides the strongest individual validation performance in this experiment, while the weighted ensemble also achieves a high validation accuracy.

**Ensemble Classification Report**

                        precision    recall  f1-score   support

     Bacterial Leaf Blight   0.95      0.90      0.92       20
            Brown Spot       1.00      0.96      0.98        27
     Healthy Rice Leaf       0.90      0.95      0.92        19
            Leaf Blast       0.97      1.00      0.98        31
            Leaf scald       1.00      1.00      1.00        23
     Narrow Brown Leaf Spot  1.00      1.00      1.00        16
            Rice Hispa       0.95      0.91      0.93        22
         Sheath Blight       0.93      0.96      0.95        28

              accuracy                           0.96       186
             macro avg       0.96      0.96      0.96       186
          weighted avg       0.96      0.96      0.96       186

**Evaluation Metrics**

The final ensemble is evaluated using several performance analysis techniques.

**Accuracy**

The overall validation accuracy of the weighted ensemble is:

                 96.24%

**Precision, Recall and F1-score**

Class-wise precision, recall and F1-score are calculated using classification_report from scikit-learn.

**Confusion Matrix**

A confusion matrix is generated to visualize the relationship between the actual rice leaf classes and the classes predicted by the ensemble.

<img width="575" height="553" alt="confusion matrix (phase -2)" src="https://github.com/user-attachments/assets/a1b66b1f-8c98-4e98-adb8-ee1c1c7abe19" />

**Multiclass ROC Analysis**

The project also generates one-vs-rest ROC curves for all eight rice leaf classes.

The Area Under the Curve (AUC) is calculated individually for each class to analyze the discriminative performance of the ensemble probabilities.

<img width="691" height="557" alt="ROC  curve" src="https://github.com/user-attachments/assets/81d88656-4ba9-4468-a423-f3e11aaadf98" />

Technologies Used

| Technology   | Purpose                                        |
| ------------ | ---------------------------------------------- |
| Python       | Main programming language                      |
| TensorFlow   | AlexNet model inference                        |
| Keras        | AlexNet model loading                          |
| PyTorch      | ResNet-18 inference                            |
| Torchvision  | ResNet-18 architecture and image preprocessing |
| NumPy        | Numerical and probability operations           |
| Scikit-learn | Accuracy and classification metrics            |
| Matplotlib   | Visualization                                  |
| Seaborn      | Confusion matrix visualization                 |

**Model Files**

The ensemble pipeline uses the trained model files generated during the project:

       rice_leaf_alexnet_model.keras

and

        rice_disease_model (2).pth

The AlexNet model is loaded using TensorFlow/Keras, while the ResNet-18 weights are loaded using PyTorch.

**Requirements**

Install the required packages using:

     pip install tensorflow torch torchvision numpy matplotlib seaborn scikit-learn

For GPU execution, install a compatible CUDA-enabled environment for the respective deep learning frameworks.

**How to Run**

1)Prepare the rice leaf validation dataset.

2)Arrange the validation images according to the documented class structure.

3)Place the trained AlexNet .keras model at the required location.

4)Place the trained ResNet-18 .pth model at the required location.

5)Open the Phase 2 notebook.

6)Update the dataset and model paths.

7)Run the notebook cells sequentially.

8)The notebook will:

- evaluate AlexNet,

- evaluate ResNet-18,

- extract their probability distributions,

- perform weighted probability fusion,

- generate the final ensemble predictions,

- calculate accuracy,

- produce the classification report,

- generate the confusion matrix, and

- generate multiclass ROC curves.

  **Results Summary**

                         Validation Accuracy

             AlexNet             69.35%
                                   │
                                   │
            ResNet-18             97.31%
                                   │
                                   │
          Weighted Ensemble       96.24%

  The experiment shows that probability-level fusion can combine predictions from different architectures into a single

  classification system. In this particular experiment, ResNet-18 achieved a higher standalone validation accuracy than the final

  weighted ensemble.

  ## **Project Structure**

        Ensemble-Learning-ResNet-AlexNet/
        │
        ├── Phase 2 Notebook
        │   └── Ensemble Learning Notebook
        │
        ├── rice_leaf_alexnet_model.keras
        │
        ├── rice_disease_model (2).pth
        │
        └── README.md

Large datasets and model files may be kept outside the repository when repository size limits make direct inclusion impractical.

## **Author**

**Subhajit Sarkar**

## **Final Note**

This repository represents the ensemble-learning phase of the rice leaf disease detection project. It focuses on combining 

predictions from two different CNN architectures and evaluating the resulting classifier through accuracy, class-wise metrics, 

confusion matrix analysis and multiclass ROC analysis.
  




