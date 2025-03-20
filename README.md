# README: ConvNeXt-Tiny for AquaMonitor JYU Dataset

## Project Overview
This project focuses on the automatic classification of aquatic macroinvertebrates using deep learning to enhance environmental biomonitoring. The **ConvNeXt-Tiny** architecture was used, achieving the best validation performance with an **83% weighted F1-score**. For comparison, the **ResNet18** model was also evaluated, reaching only **74% F1-score**.

## Objective and Relevance
Accurate monitoring of aquatic macroinvertebrates is crucial for assessing water quality and biodiversity. Manual identification is time-consuming and requires expert knowledge, making large-scale monitoring challenging. **Deep learning** can automate this process, improving classification speed and accuracy.

## Dataset Description

**AquaMonitor JYU** is a subset of the large AquaMonitor dataset containing images of aquatic macroinvertebrates.

- **Number of Classes**: 31
- **Training Set**: 40,880 images (from 1,049 individuals)
- **Validation Set**: 6,394 images (from 157 individuals)
- **Test Set**: Hidden
- **Image Format**: 256x256

### Class Distribution
![Class Distribution](https://github.com/user-attachments/assets/c0374a14-2095-474a-a035-d437c954d421)

The dataset exhibits a **significant class imbalance**. Some classes contain over **3,000 samples**, while others have fewer than **500**. This imbalance presents challenges for the model, as rare classes receive insufficient representation during training. To address this, **stronger augmentation techniques** and **dataset rebalancing** were applied.

### Class Examples
![Class Examples](https://github.com/user-attachments/assets/89e038d7-574d-494f-89b7-bbd6c2fd3d57)


## ConvNeXt-Tiny Architecture

ConvNeXt-Tiny is a modern convolutional neural network that integrates **Vision Transformer principles**, while retaining the efficiency and inductive biases of CNNs. It was chosen for its balance between traditional CNN efficiency and deep architectural improvements that enhance generalization. ConvNeXt is optimized for modern computing platforms, ensuring improved performance and classification accuracy.

### Model Configuration
- **Image Size**: 224x224
- **Number of Classes**: 31
- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss (label smoothing = 0.1)
- **Epochs**: 30
- **Batch Size**: 512
- **Max Learning Rate**: 5e-5
- **Regularization**: Cosine Annealing LR, weight decay = 1e-8

### Learning Rate Adjustment
A **differential learning rate** was applied based on the depth of the network:

A **differential learning rate** was applied based on the depth of the network. Lower layers were updated more slowly than higher layers to preserve the generalization capabilities of pre-trained weights. The base learning rate was set to **5e-5**, with a decay factor of **0.8** across different layers. Weight decay was set to **1e-8**, and the optimization was performed using the **AdamW optimizer**. A **Cosine Annealing learning rate scheduler** was employed, ensuring a gradual reduction in the learning rate over **30 epochs**, with a minimum learning rate reaching **base_lr * decay_factor^6**.

This approach allows lower layers of the model to update more slowly than higher layers, preserving the generalization capabilities of pre-trained weights.

### Model Training
- **Pretrained on ImageNet-1K**
- **Fine-tuned**: Classification head adapted to 31 classes
- **Trained in Google Colab on A100 GPU**
- **Mixed Precision (FP16)** was used for improved performance

### Data Augmentation
Two levels of augmentation were applied:

1. **General augmentation for all images**:
   - Random horizontal and vertical flipping
   - Affine transformations (translation, rotation, scaling)
   - Perspective distortions
   - Gaussian Blur
2. **Stronger augmentation for rare classes**:
   - Elastic Transform
   - Random Erasing
   - More intense contrast variations

## Results

| Architecture  | Weighted F1-score |
| ------------ | ----------------- |
| ConvNeXt-Tiny | **83%**           |
| ResNet18     | 74%               |

ConvNeXt-Tiny demonstrated **the best generalization capability**, but also showed **signs of overfitting** (99.6% accuracy on the training set).

### Confusion Matrix
![Confusion matrix](https://github.com/user-attachments/assets/d80b1456-e296-428a-8c41-f3152b0cd082)

The confusion matrix for ConvNeXt-Tiny revealed strong predictive performance across most classes. However, certain groups remained challenging, likely due to visual similarity between species. Data augmentation and oversampling contributed to improving the model’s accuracy and partially reduced overfitting by enhancing data variability and balancing class distributions.

## Conclusions

- **ConvNeXt-Tiny achieved an 83% weighted F1-score**, the highest among tested models.
- **ResNet18 performed worse**, with an F1-score of 74%.
- **Overfitting is a concern**, which can be mitigated through further regularization and augmentation techniques.

## License

This project is an academic study and is intended for research purposes.

### Model Download
The ConvNeXt-Tiny model is available at:
[Download model.pt](https://www.dropbox.com/scl/fi/vjx6fb4x82csebm60etb6/model.pt?rlkey=cgbxn3n8kyiruepkwbw9p1p7j&st=8abbh9r3&dl=0)

