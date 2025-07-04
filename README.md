# face_recognition
This repository implements a robust face recognition system using a fine-tuned ResNet-50 model, designed to perform accurately under degraded visual conditions such as blur, fog, and low light.

**Dataset link**: https://drive.google.com/file/d/1T_ygTGG4WnKbUZLyW3vDwDa1m-svRiaR/view?usp=drive_link 

# Degraded Conditions
This project implements a face recognition system trained to identify individuals even under visually degraded conditions (blur, fog, rain, low light). Built using *PyTorch* and *ResNet-50*, it includes class balancing, data augmentation, and inference with confidence scoring.

## Structure
- train_b.py: Train model using augmented & imbalanced data
- test_b.py: Predict face identity from a test image
- Face_Recognition_Model.pth: Saved best model
- Task_B/: Training & validation images (by class)
- Task_B_Dataset/: Testing images
- Robust_Face_Recognition_Technical_Summary.docx: 1-page project summary

## Setup
pip install torch torchvision scikit-learn pillow

**Train**

python train_b.py

Uses ResNet-50 pretrained on ImageNet

Weighted sampling for class imbalance

Label smoothing + dropout for generalization

Saves best model based on validation accuracy


# 📊 Training Classification Report:
                           precision    recall    f1-score   support
    macro avg                0.99        1.00      1.00       15027
    weighted avg             0.99        0.99      0.99       15027 
    accuracy                                       0.99       15027

    Training Accuracy: 98.52%

**Test**

Edit image path in test_b.py:

test_image_path = r"Task_B_Dataset/val/Class/Image.jpg"

Then run:

python test_b.py

# 📊 Output Example

📸 Image: Aishwarya_Rai_0001_blurred.jpg
🧠 Predicted Class: Aishwarya_Rai
✅ Confidence: 94.27%
