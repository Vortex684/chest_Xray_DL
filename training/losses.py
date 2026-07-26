# training/losses.py

import tensorflow as tf

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    y_true, y_pred: (batch, H, W, 8)
    """
    y_true = tf.cast(y_true, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice_score)


def bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = -(y_true * tf.math.log(y_pred) + (1-y_true) * tf.math.log(1-y_pred))
    return tf.reduce_mean(bce)


def combined_seg_loss(y_true, y_pred):
    """Dice + BCE — used for seg_output and both deep supervision heads."""
    return dice_loss(y_true, y_pred) + bce_loss(y_true, y_pred)

def weighted_focal_loss(pos_weights_tensor):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        bce = -(y_true*tf.math.log(y_pred) + (1-y_true)*tf.math.log(1-y_pred))
        p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
        focal_weight = tf.pow(1-p_t, 2.0)
        class_weights = y_true * pos_weights_tensor + (1-y_true)
        
        loss = class_weights * focal_weight * bce
        return tf.reduce_mean(loss)
    return loss_fn

def total_loss(seg_true, seg_pred, sup56_true, sup56_pred, sup112_true, sup112_pred, cls_true, cls_pred, pos_weights_tensor):
    
    loss_seg    = combined_seg_loss(seg_true, seg_pred)
    loss_sup56  = combined_seg_loss(sup56_true, sup56_pred)
    loss_sup112 = combined_seg_loss(sup112_true, sup112_pred)
    loss_cls    = weighted_focal_loss(pos_weights_tensor)(cls_true, cls_pred)
    
    total = (0.5 * loss_seg + 
             0.2 * loss_sup56 + 
             0.3 * loss_sup112 + 
             1.0 * loss_cls)
    
    return total