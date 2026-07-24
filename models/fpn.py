# models/fpn.py

import tensorflow as tf
from keras import layers

def fpn_block(C2, C3, C4, aspp_output, num_filters=256):
    """
    Feature Pyramid Network — top-down semantic fusion.
    
    Inputs:
        C2:          28x28x40    encoder skip — fine spatial detail
        C3:          14x14x112   encoder skip — medium spatial detail
        C4:          7x7x192     encoder skip — coarse spatial detail
        aspp_output: 7x7x256    rich multi-scale semantics from ASPP
    
    Output:
        P2: 28x28x256  — semantics + fine spatial detail → feeds decoder
    """
    
    # --- Lateral connections ---
    # Align encoder channels to num_filters (256) using 1x1 conv
    # No spatial reasoning — pure channel alignment
    
    C4_aligned = layers.Conv2D(num_filters, 1, padding='same', use_bias=False)(C4)
    C4_aligned = layers.BatchNormalization()(C4_aligned)
    
    C3_aligned = layers.Conv2D(num_filters, 1, padding='same', use_bias=False)(C3)
    C3_aligned = layers.BatchNormalization()(C3_aligned)
    
    C2_aligned = layers.Conv2D(num_filters, 1, padding='same', use_bias=False)(C2)
    C2_aligned = layers.BatchNormalization()(C2_aligned)
    
    # --- Top-down pathway ---
    
    # Step 1: merge ASPP with C4 at 7x7
    P4 = layers.Add()([aspp_output, C4_aligned])  # 7x7x256
    P4 = layers.Conv2D(num_filters, 3, padding='same', use_bias=False)(P4)
    P4 = layers.BatchNormalization()(P4)
    P4 = layers.Activation('relu')(P4)
    
    # Step 2: upsample P4 to 14x14, merge with C3
    P4_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(P4)  # 14x14x256
    P3 = layers.Add()([P4_up, C3_aligned])         # 14x14x256
    P3 = layers.Conv2D(num_filters, 3, padding='same', use_bias=False)(P3)
    P3 = layers.BatchNormalization()(P3)
    P3 = layers.Activation('relu')(P3)
    
    # Step 3: upsample P3 to 28x28, merge with C2
    P3_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(P3)  # 28x28x256
    P2 = layers.Add()([P3_up, C2_aligned])          # 28x28x256
    P2 = layers.Conv2D(num_filters, 3, padding='same', use_bias=False)(P2)
    P2 = layers.BatchNormalization()(P2)
    P2 = layers.Activation('relu')(P2)
    
    return P2  # 28x28x256 → feeds decoder