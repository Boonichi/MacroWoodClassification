# Macroscopic Wood Classification

## Statement

## Description

## Pipeline (Developed)
- Data Split: Stratified K-Fold (num_folds = 5) 
- Data Augmentation:
    - Train Dataset:
        - Resize: 224x224
        - Color_jitter : 0.4
        - Vertical flip : 0.5
        - Horizontal flip: 0.5
        - Interpolation: bicubic
        - Random Erase params:
            - reprob: 0.25
            - remode: pixel
            - recount: 1
        - RandomCrop (img_size > 32): padding = 4
        - Normalization (Mean, std): ((0.4875, 0.4194, 0.3976),(0.0332, 0.0276, 0.0245))
    - Validation Dataset:
        - Resize: 224x224
        - Interpolation: bicubic
        - Normalize (Mean, std)
- Sampler techniques:
    - Train Dataset: DistributedSampler
    - Val Dataset: SequentialSampler
- Criterion (Loss): PolyLoss, LabelSmoothingCrossEntropy, CrossEntropy 
- Logging: TensorBoard, Wandb
- Callbacks: Early Stopping, LRMonitor
- Optimizers: Adamw, Nadam, radam (easy to add more options) 
- Simple Models (Resnet, VGG, densenet, inception, ...)

### Undeveloped
- Models: Transformer (Attentionhead)
- Processes: Finetune, predict
- Augmentation: Mixup
- Optimization: LR/Layer decay (Cosine Scheduler), ModelEMA, LossScaler
- Save Model
