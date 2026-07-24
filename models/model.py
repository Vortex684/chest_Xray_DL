
import tensorflow as tf
from keras import layers, Model
from models.encoder import build_encoder
from models.aspp import aspp_block
from models.fpn import fpn_block
from models.decoder import decoder_block

def build_full_model(input_shape=(224, 224, 3), num_seg_classes=8, num_all_classes=14, encoder_trainable=False):
    
    inputs = layers.Input(shape=input_shape)
    
    encoder = build_encoder(input_shape=input_shape, trainable=encoder_trainable)
    C2, C3, C4, C5 = encoder(inputs)
    
    aspp_out = aspp_block(C5)
    
    P2 = fpn_block(C2, C3, C4, aspp_out)
    
    seg_output, deep_sup_56, deep_sup_112 = decoder_block(P2, num_classes=num_seg_classes)
    
    # Classification head — reuses C5 directly
    cls = layers.GlobalAveragePooling2D()(C5)
    cls = layers.Dropout(0.3)(cls, training=True)
    cls = layers.Dense(128, activation='relu')(cls)
    cls = layers.Dropout(0.3)(cls, training=True)
    cls_output = layers.Dense(num_all_classes, activation='sigmoid', name='cls_output')(cls)
    
    model = Model(
        inputs=inputs,
        outputs=[seg_output, deep_sup_56, deep_sup_112, cls_output],
        name='chest_segmentation_model'
    )
    
    return model