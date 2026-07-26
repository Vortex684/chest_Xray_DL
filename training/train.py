# training/train.py

import pandas as pd
import numpy as np
import tensorflow as tf
import os

from data.preprocessing import load_bbox_data, encode_labels, ALL_DISEASES
from data.generator import ChestXrayGenerator
from models.model import build_full_model
from training.losses import combined_seg_loss, weighted_focal_loss

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Training on {len(gpus)} GPUs using MirroredStrategy")
else:
    strategy = tf.distribute.get_strategy()

# ---- Step 1: Set up paths (Kaggle) ----
BASE = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'

# ---- Step 2: Build path_lookup — same as your old project ----
image_dirs = [f'{BASE}/images_{str(i).zfill(3)}/images' for i in range(1, 13)]
path_lookup = {}
for folder in image_dirs:
    for fname in os.listdir(folder):
        path_lookup[fname] = os.path.join(folder, fname)

# ---- Step 3: Load main CSV and encode labels ----
df = pd.read_csv(f'{BASE}/Data_Entry_2017.csv')
df['label_vector'] = df['Finding Labels'].apply(encode_labels)

# ---- Step 4: Split into train/test using NIH's official split ----
with open(f'{BASE}/train_val_list.txt') as f:
    train_files = set(f.read().splitlines())
with open(f'{BASE}/test_list.txt') as f:
    test_files = set(f.read().splitlines())

train_df = df[df['Image Index'].isin(train_files)].reset_index(drop=True)
val_df   = df[df['Image Index'].isin(test_files)].reset_index(drop=True)

# ---- Step 5: Load bbox annotations ----
bbox_lookup = load_bbox_data(BASE)

# ---- Step 6: NOW pos_weights can be computed ----
labels_array = np.array(train_df['label_vector'].tolist())
pos_weights = (labels_array==0).sum(axis=0) // (labels_array==1).sum(axis=0)
pos_weights_tensor = tf.constant(pos_weights, dtype=tf.float32)

# ---- Step 7: Build generators ----
train_gen = ChestXrayGenerator(train_df, path_lookup, bbox_lookup, augment=True)
val_gen   = ChestXrayGenerator(val_df,   path_lookup, bbox_lookup, augment=False)

# ---- Step 8: Build and compile model ----
with strategy.scope():
    model = build_full_model(encoder_trainable=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'seg_output':   combined_seg_loss,
            'deep_sup_112': combined_seg_loss,
            'deep_sup_56':  combined_seg_loss,
            'cls_output':   weighted_focal_loss(pos_weights_tensor)
        },
        loss_weights={
            'seg_output':   0.5,
            'deep_sup_112': 0.3,
            'deep_sup_56':  0.2,
            'cls_output':   1.0
        }
)