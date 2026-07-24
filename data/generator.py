import numpy as np
import tensorflow as tf
from PIL import Image
from keras.utils import Sequence 
from data.preprocessing import create_mask, encode_labels

augment_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05)
])

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

        images       = []
        masks        = []
        label_vectors = []

        for _, row in batch.iterrows():
            # Image
            path = self.path_lookup[row['Image Index']]
            img  = Image.open(path).convert('RGB')
            img  = img.resize((self.img_size, self.img_size))
            img  = np.array(img) / 255.0
            images.append(img)

            # Mask
            mask = create_mask(row['Image Index'], self.bbox_lookup, self.img_size)
            masks.append(mask)

            # Label vector
            label_vectors.append(row['label_vector'])

        images = np.array(images, dtype=np.float32)
        masks  = np.array(masks,  dtype=np.float32)

        if self.augment:
            images = augment_layer(images, training=True).numpy()

        return images, masks, np.array(label_vectors, dtype=np.float32)