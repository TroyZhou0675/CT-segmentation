from tensorflow.keras.applications import ResNet50

# 加载 ResNet50 模型
# include_top=False 表示不包括顶部的全连接层
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 打印模型中所有层的名称
print("ResNet50 模型中的所有层名:")
for layer in base_model.layers:
    print(layer.name)