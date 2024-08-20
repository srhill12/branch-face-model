
# Branch-Face-Model

This repository contains a project focused on building and evaluating a multi-output model to classify user identity, pose, facial expression, and eye state from a facial dataset. The approach utilizes a custom multi-output neural network architecture designed to handle multiple prediction tasks simultaneously.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Insights and Analysis](#insights-and-analysis)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to implement a multi-output deep learning model that predicts multiple labels from facial images. Specifically, the model predicts:

- **User Identity**
- **Pose**
- **Expression**
- **Eye State**

The project is designed to demonstrate the ability of a single model to handle multiple related tasks, which is useful in applications where facial recognition and expression analysis are required simultaneously.

## Installation

To use the code in this repository, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/srhill12/branch-face-model.git
    ```

2. Navigate to the project directory:

    ```bash
    cd branch-face-model
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train and evaluate the model, run the notebook `branching-faces.ipynb`. The notebook is organized into the following sections:

1. **Data Loading and Preprocessing**: Load and preprocess the facial image dataset, including splitting into training and test sets.
2. **Model Definition**: Define a custom multi-output model architecture using Keras.
3. **Training**: Train the model on the dataset with specified hyperparameters.
4. **Evaluation**: Evaluate the model's performance on test data and analyze accuracy for each output.

## Model Architecture

The model is a custom multi-output neural network that handles multiple prediction tasks. The architecture is as follows:

- Shared layers process the input image.
- Separate output layers for each prediction task (user ID, pose, expression, eyes).
- Each output layer is trained to predict its respective label.

This architecture allows for the simultaneous learning of shared features and specific features relevant to each task.

## Training and Evaluation

### Training

The model is trained using the following parameters:

- **Epochs**: 10 (adjustable based on needs)
- **Batch Size**: 32 (adjustable based on available memory)
- **Validation Split**: 20% of the training data

### Evaluation

The model is evaluated on the test dataset, and the accuracy for each prediction task is reported. The results are organized as follows:

- **User ID Accuracy**
- **Pose Accuracy**
- **Expression Accuracy**
- **Eyes Accuracy**

## Results

The results of the model evaluation indicate the accuracy achieved for each task:

- **User ID Accuracy**: `0.9038`
- **Pose Accuracy**: `0.7307`
- **Expression Accuracy**: `0.1580`
- **Eyes Accuracy**: `0.7981`

## Insights and Analysis

### Model Performance

The model performs well on the User ID and Eye State tasks, with accuracies of 90.38% and 79.81%, respectively. However, the performance on the Expression task is notably lower, with an accuracy of only 15.80%. This disparity suggests that the model might be struggling with the complexity of recognizing expressions, possibly due to insufficient data or model capacity for this task.

### Multi-Task Learning

The shared layers allow the model to learn common features, which is beneficial for tasks like User ID and Pose, where the facial structure plays a significant role. However, the Expression task might require more specific features that are not adequately captured by the shared layers. Further experimentation with dedicated layers for more complex tasks might improve performance.

### Next Steps

To improve the model, consider the following approaches:

- **Data Augmentation**: Apply augmentation techniques to increase the variety of expressions in the dataset.
- **Hyperparameter Tuning**: Experiment with different architectures and learning rates.
- **Transfer Learning**: Utilize a pre-trained model to leverage features learned from a large facial dataset.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

