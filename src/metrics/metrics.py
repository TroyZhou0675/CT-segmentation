import tensorflow as tf

def multiclass_dice(y_true,y_pred,smooth = 1e-6,ignore_background = True):

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

    return tf.reduce_mean(dice_per_class)

'''''
Iou的函数实现:用于训练实时监测
'''''
def multiclss_soft_iou(y_true,y_pred,smooth = 1e-6,ignore_background = True):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    n = tf.shape(y_true)[0]
    c = tf.shape(y_true)[-1]

    y_true = tf.reshape(y_true,[n,-1,c])
    y_pred = tf.reshape(y_pred,[n,-1,c])

    intersection = tf.reduce_sum(y_true * y_pred,axis = 1)
    denom = tf.reduce_sum(y_true + y_pred - y_true * y_pred,axis=1)
    iou = (intersection  + smooth)/(denom + smooth)

    iou_per_class = tf.reduce_mean(iou,axis = 0)
    if ignore_background is True:
        iou_per_class = iou_per_class[1:]

    return tf.reduce_mean(iou_per_class)

'''''
严格按照metrics接口定义的MeanIou,用于最终模型测评
（未测试）
'''''
class MeanIouFromProbs(tf.keras.metrics.Metric):
    def __init__(self,num_classes, name='mean_iou_from_probs', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.miou = tf.keras.metrics.MeanIoU(num_classes= num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_idx = tf.argmax(y_true,axis = -1)
        y_pred_idx = tf.argmax(y_pred,axis = -1)
        return self.miou.update_state(y_true_idx,y_pred_idx,sample_weight=None)

    def result(self):
        return self.miou.result()