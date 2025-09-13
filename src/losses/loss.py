import tensorflow as tf

def dice_loss(y_true, y_pred, smooth=1e-6,ignore_background = True):
    """
    多分类 Dice Loss
    """
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    n = tf.shape(y_true)[0]
    c = tf.shape(y_true)[-1]

    y_true = tf.reshape(y_true,[n,-1,c])
    y_pred = tf.reshape(y_pred,[n,-1,c])

    intersection = tf.reduce_sum(y_true*y_pred,axis = 1)
    denom = tf.reduce_sum(y_true+y_pred,axis=1)
    dice = (2*intersection  + smooth)/(denom + smooth)

    dice_per_class = tf.reduce_mean(dice,axis = 0)
    if ignore_background is True:
        dice_per_class = dice_per_class[1:]

    loss = 1 - tf.reduce_mean(dice_per_class)
    return loss


def WCE_loss(weights):

    w = tf.constant(weights,dtype = tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        
        sample_weights = tf.reduce_sum(y_true * w, axis=-1)
        weighted_ce = ce * sample_weights
        
        cce_loss = tf.reduce_mean(weighted_ce)

        return cce_loss
    
    return loss


def Combined_loss(weights,weight_of_dice,ignore_the_back = True):
    wce_loss = WCE_loss(weights)
    def loss(y_true,y_pred):
        return weight_of_dice*dice_loss(y_true,y_pred,ignore_background = ignore_the_back) + (1 - weight_of_dice)*wce_loss(y_true,y_pred)
    return loss