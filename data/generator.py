# data/generator.py

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.utils import Sequence
from data.preprocessing import create_mask

# Same augmentation as your old project
augment_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05)
])


def downsample_mask(mask, size):
    """
    Takes a 224x224x8 mask and resizes it to size x size x 8.
    Used to create ground truth for deep supervision heads
    at 56x56 and 112x112.
    
    method='nearest' keeps values as clean 0s and 1s —
    bilinear would blend edges into fractional values like 0.3, 0.7
    which doesn't make sense for a binary mask.
    """
    mask_tensor = tf.convert_to_tensor(mask)[tf.newaxis, ...]  # add batch dim: (1,224,224,8)
    resized = tf.image.resize(mask_tensor, (size, size), method='nearest')
    return resized[0].numpy()  # remove batch dim: (size,size,8)


class ChestXrayGenerator(Sequence):

    def __init__(self, dataframe, path_lookup, bbox_lookup,
                 batch_size=16, img_size=224, augment=False):
        self.df          = dataframe
        self.path_lookup = path_lookup
        self.bbox_lookup = bbox_lookup
        self.batch_size  = batch_size
        self.img_size    = img_size
        self.augment     = augment

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        images        = []
        masks_224     = []
        masks_112     = []
        masks_56      = []
        label_vectors = []

        for _, row in batch.iterrows():
            
            # ---- Image ----
            path = self.path_lookup[row['Image Index']]
            img  = Image.open(path).convert('RGB')
            img  = img.resize((self.img_size, self.img_size))
            img  = np.array(img) / 255.0
            images.append(img)

            # ---- Mask at full resolution ----
            mask_224 = create_mask(row['Image Index'], self.bbox_lookup, self.img_size)
            masks_224.append(mask_224)

            # ---- Downsample for deep supervision heads ----
            masks_112.append(downsample_mask(mask_224, 112))
            masks_56.append(downsample_mask(mask_224, 56))

            # ---- Classification label vector ----
            label_vectors.append(row['label_vector'])

        images    = np.array(images, dtype=np.float32)
        masks_224 = np.array(masks_224, dtype=np.float32)
        masks_112 = np.array(masks_112, dtype=np.float32)
        masks_56  = np.array(masks_56,  dtype=np.float32)
        label_vectors = np.array(label_vectors, dtype=np.float32)

        if self.augment:
            images = augment_layer(images, training=True).numpy()

        # Dict keys MUST match your model's output layer names exactly
        return images, {
            'seg_output':   masks_224,
            'deep_sup_112': masks_112,
            'deep_sup_56':  masks_56,
            'cls_output':   label_vectors
        }