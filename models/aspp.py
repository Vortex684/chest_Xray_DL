import tensorflow as tf
from keras import layers, Model

def aspp_block(x, num_filters=256):
    """
    Atrous Spatial Pyramid Pooling.
    
    Input:  7x7x1280  (C5 from encoder)
    Output: 7x7x256
    
    Five parallel branches:
    1. dilation=1  → local features
    2. dilation=2  → slightly wider
    3. dilation=4  → medium scale
    4. dilation=6  → large scale (not 8 — spatial size is only 7x7)
    5. GlobalAvgPool → global context
    """
    # 1x1 Conv2D looks at one pixel position at a time. It takes that pixel's 1280
    # channel values and computes a weighted combination to produce 256 output values.
    b1=layers.Conv2D(num_filters,1,padding='same',use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)
    
        # Branch 2 — dilation=2
    b2 = layers.Conv2D(num_filters, 3, padding='same', dilation_rate=2, use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)
    
    # Branch 3 — dilation=4
    b3 = layers.Conv2D(num_filters, 3, padding='same', dilation_rate=4, use_bias=False)(x)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)
    
    # Branch 4 — dilation=6
    # NOT dilation=8 — on a 7x7 map dilation=8 would sample outside boundaries
    b4 = layers.Conv2D(num_filters, 3, padding='same', dilation_rate=6, use_bias=False)(x)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)
    
    # Branch 5 — global average pooling
    b5 = layers.GlobalAveragePooling2D()(x)
    b5 = layers.Reshape((1, 1, x.shape[-1]))(b5)
    b5 = layers.Conv2D(num_filters, 1, padding='same', use_bias=False)(b5)
    b5 = layers.BatchNormalization()(b5)
    b5 = layers.Activation('relu')(b5)
    b5 = layers.UpSampling2D(size=(7, 7), interpolation='bilinear')(b5)
    
    # Concat all five branches
    x = layers.Concatenate()([b1, b2, b3, b4, b5])  # 7x7x1280                                                              
    
    # Compress back to num_filters
    x = layers.Conv2D(num_filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x