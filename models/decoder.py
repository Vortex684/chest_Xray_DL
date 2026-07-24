
import tensorflow as tf
from keras import layers

def conv_bn_relu(x, filters, kernel_size=3):
    """Standard conv block — used repeatedly in decoder."""
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def decoder_block(P2, num_classes=8, dropout_rate=0.3):
    """
    U-Net style decoder with deep supervision.
    
    Input:  P2 → 28x28x256  (from FPN)
    Output: 
        seg_output:  224x224x8  final segmentation mask
        deep_sup_56: 56x56x8   intermediate supervision
        deep_sup_112:112x112x8  intermediate supervision
    """
    
    # ── Stage 1: 28x28 → 56x56 ──────────────────────────────
    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(P2)  # 56x56x256
    x = conv_bn_relu(x, 128)                                            # 56x56x128
    x = layers.Dropout(dropout_rate)(x, training=True)                 # MC Dropout
    
    # Deep supervision head at 56x56
    deep_sup_56 = layers.Conv2D(num_classes, 1, activation='sigmoid', name='deep_sup_56')(x)                  # 56x56x8
    
    # ── Stage 2: 56x56 → 112x112 ────────────────────────────
    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)   # 112x112x128
    x = conv_bn_relu(x, 64)                                             # 112x112x64
    x = layers.Dropout(dropout_rate)(x, training=True)
    
    # Deep supervision head at 112x112
    deep_sup_112 = layers.Conv2D(num_classes, 1, activation='sigmoid', name='deep_sup_112')(x)                # 112x112x8
    
    # ── Stage 3: 112x112 → 224x224 ──────────────────────────
    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)   # 224x224x64
    x = conv_bn_relu(x, 32)                                             # 224x224x32
    x = layers.Dropout(dropout_rate)(x, training=True)
    
    # Final segmentation output
    seg_output = layers.Conv2D(num_classes, 1, activation='sigmoid', name='seg_output')(x)                    # 224x224x8
    
    return seg_output, deep_sup_56, deep_sup_112