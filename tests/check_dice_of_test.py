import sys
import numpy as np
import tensorflow as tf
sys.path.append(r'D:/0-MyDoc/DeepLearning/MIS/CT_SEG/')
from src import *


IMG_SIZE = (256, 256, 3)
BATCH_SIZE = 6
NUM_CLASSES = 4
model_path = 'D:/0-MyDoc/DeepLearning/MIS/CT_SEG/experiments/2025-9-11-1/best_unet.h5'
model = Nestnet(input_shape=IMG_SIZE,freeze_encoder=True,classes=NUM_CLASSES)
model.load_weights(model_path)

image_partition, mask_partition = get_partitions(
    images_dir='./data/processed/images/',
    masks_dir='./data/processed/masks/'
)

val_gen = CustomDataGenerator(
    image_filenames=image_partition['test'],
    mask_filenames=mask_partition['test'],
    batch_size=BATCH_SIZE,
    dim=IMG_SIZE,
    n_classes=NUM_CLASSES,
    augment=False,
    shuffle=False
)

dice_total = 0
for i, (images_batch, true_masks_batch) in enumerate(val_gen):
    # Predict on the current batch
    results_batch = model.predict_on_batch(images_batch)
    
    # Use argmax to convert probabilities to class labels
    predicted_masks_batch = np.argmax(results_batch, axis=-1)
    true_masks_labels_batch = np.argmax(true_masks_batch, axis=-1)
    
    # Get the current batch size (can be smaller for the last batch)
    batch_size_current = true_masks_batch.shape[0]
    
    # Iterate through each image in the batch to calculate the metric

    for j in range(batch_size_current):
        dice_total += multiclass_dice(true_masks_labels_batch[j], predicted_masks_batch[j], NUM_CLASSES)
        num_samples += 1

# 计算最终的平均 Dice 分数
if num_samples > 0:
    dice_average = dice_total / num_samples
    print(f"在测试集上的平均 Dice 系数: {dice_average.numpy()}")
else:
    print("没有样本被处理。")