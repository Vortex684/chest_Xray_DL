import tensorflow as tf
from keras.applications import EfficientNetB0
from keras import Model

base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))

for layer in base.layers:
    if hasattr(layer, 'output'): # skip layers that don't expose an output tensor
        try:
            shape = layer.output.shape
            # We only care about spatial sizes 28, 14, 7
            # These are our C2, C3, C4 skip connection points
            # 28x28 = fine detail, 14x14 = medium, 7x7 = semantic
            if len(shape) == 4 and shape[1] in [28, 14, 7]:
                print(layer.name, shape)
        except:
            continue


def build_encoder(input_shape=(224, 224, 3), trainable=False):
    
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    base.trainable = trainable
    
    C2 = base.get_layer('block3b_add').output      # 28x28x40
    C3 = base.get_layer('block5c_add').output      # 14x14x112
    C4 = base.get_layer('block6d_add').output      # 7x7x192
    C5 = base.get_layer('top_activation').output   # 7x7x1280
    
    encoder = Model(
        inputs=base.input,
        outputs=[C2, C3, C4, C5],
        name='encoder'
    )
    
    return encoder