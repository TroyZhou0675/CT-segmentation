import sys

from numpy import empty
sys.path.append(r'D:/0-MyDoc/DeepLearning/MIS/CT_SEG/')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src import *
import os

config_path = './configs/2025-9-11-2.yaml'

abs_path = os.path.abspath(config_path)
config = get_yaml(abs_path)

EXPERIMENT_PATH = config['experiment_path']
IMG_SIZE = tuple(config['img_size'])
NUM_CLASSES = config["num_classes"]
MODEL_PATH = EXPERIMENT_PATH + '/best_unet.h5'
WEIGHT_IN_CE = config["weight_in_ce"]
WEIGHT_OF_DICE = config["weight_of_dice"]
IGNORE_BACKGROUND = config["ignore_background"]
BATCH_SIZE = config['batch_size']
EPOCHS = config["epochs"]
LR = config["learning_rate"]
LOAD_WEIGHT_PATH = config["load_weight_path"]

image_partition, mask_partition = get_partitions(
    images_dir='./data/processed/images/',
    masks_dir='./data/processed/masks/'
)

train_gen = CustomDataGenerator(
    image_filenames=image_partition['train'],
    mask_filenames=mask_partition['train'],
    batch_size=BATCH_SIZE,
    dim=IMG_SIZE,
    n_classes=NUM_CLASSES,
    augment=True,
    shuffle=True
)

val_gen = CustomDataGenerator(
    image_filenames=image_partition['val'],
    mask_filenames=mask_partition['val'],
    batch_size=BATCH_SIZE,
    dim=IMG_SIZE,
    n_classes=NUM_CLASSES,
    augment=False,
    shuffle=False
)

loss_fn = Combined_loss(
    weights=WEIGHT_IN_CE,
    weight_of_dice=WEIGHT_OF_DICE,
    ignore_the_back=IGNORE_BACKGROUND
)

model = Nestnet(input_shape=IMG_SIZE,freeze_encoder=False,classes=NUM_CLASSES)
model.compile(optimizer = Adam(learning_rate = float(LR)), loss = loss_fn, metrics = [multiclss_soft_iou])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

if LOAD_WEIGHT_PATH is None:
    print("Train From Scratch")
else:
    model.load_weights(LOAD_WEIGHT_PATH + '/best_unet.h5')
    print(f"Load weight {LOAD_WEIGHT_PATH}")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

model.save(EXPERIMENT_PATH + "/final_unet.h5")
plot_history(history,EXPERIMENT_PATH)