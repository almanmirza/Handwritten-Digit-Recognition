# Handwritten-Digit-Recognition
Alman Ahmad
February 8, 2025

1 Introduction
This report provides a comprehensive overview of the improvements made to the handwritten
digit classification model. It details the transition from a simple fully connected
neural network to a more advanced Convolutional Neural Network (CNN) and documents
the hyperparameter tuning process, architectural comparisons, and final evaluation insights.

2 Model Improvements
Initially, the model was a basic feedforward neural network with the following structure:
• Input Layer: Flattened 28x28 grayscale images (138 neurons)
• Hidden Layer: 128 neurons with ReLU activation
• Output Layer: 10 neurons with Softmax activation
While this model performed decently, it lacked the ability to extract spatial hierarchies
in images effectively. To improve performance, the following enhancements were implemented:

2.1 Transition to CNN Architecture
• Convolutional Layers: Extract spatial features, improving recognition ability.
• Batch Normalization: Stabilizes learning and accelerates training.
• Dropout Layers: Prevents overfitting by randomly disabling neurons during training.
• Max-Pooling Layers: Reduces spatial dimensions while retaining key features.
• L2 Regularization: Adds weight penalties to reduce complexity and overfitting.

2.2 Final CNN Model Architecture
1. Conv2D (32 filters, 3x3, ReLU, L2 Regularization) + Batch Normalization + Max-
Pooling + Dropout (0.25)
2. Conv2D (64 filters, 3x3, ReLU, L2 Regularization) + Batch Normalization + Max-
Pooling + Dropout (0.25)
3. Fully Connected Layer (128 neurons, ReLU, L2 Regularization) + Batch Normalization
+ Dropout (0.5)
4. Output Layer (10 neurons, Softmax activation)
This new architecture significantly improved the model’s ability to learn and generalize
patterns from digit images.

3 Hyperparameter Tuning Process
To optimize the model’s performance, multiple hyperparameters were fine-tuned:
• Learning Rate: 0.001 provided the best balance between convergence speed and
stability.
• Batch Size: 64 offered the best trade-off between computational efficiency and
model accuracy.
• Number of Filters: 32 and 64 provided the best feature extraction without excessive
computation.
• Dropout Rate: 0.25 in convolutional layers and 0.5 in the fully connected layer
minimized overfitting.
Each combination was tested using validation accuracy and loss curves. The final chosen
parameters provided the best generalization on unseen data.

4 Model Comparisons
4.1 Baseline Fully Connected Model
• Test Accuracy: ˜97%
• Key Limitation: Struggled with misclassifications due to lack of spatial feature
extraction.
4.2 Improved CNN Model
• Test Accuracy: ˜98%
• Key Strengths: Better feature extraction, improved generalization, reduced misclassification.
Overall, the CNN outperformed the fully connected model by a significant margin, demonstrating
the advantages of convolutional layers for image classification tasks.

5 Final Evaluation Metrics & Insights
To thoroughly evaluate the final model, we analyzed:
5.1 Confusion Matrix
• Showed minimal misclassifications, primarily among visually similar digits (e.g., 3
& 8, 4 & 9).
• Higher accuracy across all classes compared to the initial model.
5.2 Precision-Recall & ROC Curve Analysis
• Precision and recall were near 99% for all classes.
• ROC curves for each class showed high AUC values ( 0.999 for most classes), confirming
strong classifier performance.
5.3 Key Insights
• Regularization & Dropout were essential: They significantly improved generalization
and reduced overfitting.
• CNNs excel in spatial learning: The ability to capture local pixel relationships
was crucial in boosting accuracy.
• Hyperparameter tuning played a key role: Optimizing learning rate, dropout,
and batch size helped achieve the final accuracy of 98%.

6 Conclusion & Future Work
The transition to a CNN architecture drastically improved model performance, reducing
misclassification and increasing robustness. Future enhancements could include:
• Implementing data augmentation to further improve generalization.
• Exploring transfer learning using pre-trained CNNs like VGG or ResNet.
• Optimizing the model using quantization or pruning for deployment on edge
devices.
This project successfully demonstrated how architectural improvements and careful hyperparameter
tuning can enhance digit classification accuracy.
