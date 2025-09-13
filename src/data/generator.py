import numpy as np
import tensorflow as tf
import math
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import AffineTransform, warp
from scipy.ndimage import map_coordinates, gaussian_filter
import tensorflow as tf
import math

class EnhancedAugmentor:
    """
    一个功能更全面的图像和掩码增强器。
    在原有随机仿射+弹性形变的基础上，增加了：
    - 水平/垂直翻转
    - 亮度/对比度调整
    - 高斯噪声
    - 高斯模糊
    - 随机擦除 (Cutout)
    """
    def __init__(self,
                 # --- 原有几何参数 ---
                 max_rotate_deg=15,
                 max_shift_ratio=0.05,
                 max_zoom_ratio=0.10,
                 max_shear_deg=10,
                 elastic_alpha=34,
                 elastic_sigma=4,
                 elastic_prob=0.5,

                 # --- 新增翻转参数 ---
                 h_flip_prob=0.5,
                 v_flip_prob=0.0,  # CT轴状位通常不建议垂直翻转，可按需开启

                 # --- 新增强度/噪声/模糊参数 (仅作用于图像) ---
                 brightness_contrast_prob=0.5,
                 brightness_limit=0.1,  # 亮度变化范围 (-0.1, 0.1)
                 contrast_limit=0.1,    # 对比度变化范围 (-0.1, 0.1)

                 noise_prob=0.3,
                 noise_std_dev_max=20.0, # 添加高斯噪声的标准差最大值 (基于0-255像素范围)

                 blur_prob=0.3,
                 blur_sigma_max=1.0,     # 高斯模糊的sigma最大值

                 # --- 新增遮挡参数 ---
                 cutout_prob=0.3,
                 cutout_max_size_ratio=0.1, # 遮挡块最大尺寸占图像尺寸的比例
                 cutout_fill_value=0        # 遮挡区域的填充值
                ):

        # --- 几何参数 ---
        self.max_rotate_deg = max_rotate_deg
        self.max_shift_ratio = max_shift_ratio
        self.max_zoom_ratio = max_zoom_ratio
        self.max_shear_deg = max_shear_deg
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob

        # --- 强度/噪声/模糊参数 ---
        self.brightness_contrast_prob = brightness_contrast_prob
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.noise_prob = noise_prob
        self.noise_std_dev_max = noise_std_dev_max
        self.blur_prob = blur_prob
        self.blur_sigma_max = blur_sigma_max
        
        # --- 遮挡参数 ---
        self.cutout_prob = cutout_prob
        self.cutout_max_size_ratio = cutout_max_size_ratio
        self.cutout_fill_value = cutout_fill_value


    def _apply_intensity_augmentations(self, img):
        """私有方法：应用所有仅针对图像的强度类增强"""
        # 保存原始数据类型，以便最后恢复
        original_dtype = img.dtype
        img = img.astype(np.float32)
        
        # --- 3) 亮度与对比度 ---
        if np.random.rand() < self.brightness_contrast_prob:
            # alpha: 对比度, beta: 亮度
            alpha = 1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit)
            beta = np.random.uniform(-self.brightness_limit, self.brightness_limit) * 255 # 假设范围0-255
            img = img * alpha + beta
            img = np.clip(img, 0, 255) # 裁剪到有效范围

        # --- 4) 高斯噪声 ---
        if np.random.rand() < self.noise_prob:
            std_dev = np.random.uniform(0, self.noise_std_dev_max)
            noise = np.random.normal(0, std_dev, img.shape)
            img += noise
            img = np.clip(img, 0, 255)

        # --- 5) 高斯模糊 ---
        if np.random.rand() < self.blur_prob:
            sigma = np.random.uniform(0, self.blur_sigma_max)
            # 对多通道图像的每个通道分别应用模糊
            if img.ndim == 3:
                for c in range(img.shape[-1]):
                    img[..., c] = gaussian_filter(img[..., c], sigma)
            else:
                img = gaussian_filter(img, sigma)
        
        return img.astype(original_dtype)

    def _apply_cutout(self, img, mask):
        """私有方法：应用随机擦除 (Cutout)"""
        if np.random.rand() < self.cutout_prob:
            H, W = img.shape[:2]
            
            cutout_h = int(np.random.uniform(0.1, self.cutout_max_size_ratio) * H)
            cutout_w = int(np.random.uniform(0.1, self.cutout_max_size_ratio) * W)
            
            y1 = np.random.randint(0, H - cutout_h + 1)
            x1 = np.random.randint(0, W - cutout_w + 1)
            y2, x2 = y1 + cutout_h, x1 + cutout_w
            
            img[y1:y2, x1:x2] = self.cutout_fill_value
            # 同时在mask上进行擦除，填充值为0
            mask[y1:y2, x1:x2] = 0
            
        return img, mask

    def __call__(self, img, mask):
        """
        增强流程:
        1. 翻转 (几何)
        2. 仿射+弹性形变 (几何)
        3. 亮度/对比度/噪声/模糊 (强度，仅图像)
        4. 随机擦除 (Cutout)
        """
        img_out, mask_out = img.copy(), mask.copy()
        H, W = img_out.shape[:2]

        # —— 1) 随机翻转 —— #
        if np.random.rand() < self.h_flip_prob:
            img_out = np.fliplr(img_out)
            mask_out = np.fliplr(mask_out)
        if np.random.rand() < self.v_flip_prob:
            img_out = np.flipud(img_out)
            mask_out = np.flipud(mask_out)

        # —— 2) 仿射 + 弹性形变 (与您原有的逻辑一致) —— #
        # 仿射参数
        angle = np.deg2rad(np.random.uniform(-self.max_rotate_deg, self.max_rotate_deg))
        shear = np.deg2rad(np.random.uniform(-self.max_shear_deg, self.max_shear_deg))
        scale = 1.0 + np.random.uniform(-self.max_zoom_ratio, self.max_zoom_ratio)
        tx = np.random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * W
        ty = np.random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * H
        tform = AffineTransform(scale=(scale, scale), rotation=angle, shear=shear, translation=(tx, ty))
        
        img_affine = warp(img_out, tform.inverse, order=1, mode='constant', cval=0, preserve_range=True)
        if mask_out.ndim == 2:
            mask_affine = warp(mask_out, tform.inverse, order=0, mode='constant', cval=0, preserve_range=True)
        else:
            mask_affine = np.stack([
                warp(mask_out[..., c], tform.inverse, order=0, mode='constant', cval=0, preserve_range=True)
                for c in range(mask_out.shape[-1])
            ], axis=-1)

        # 弹性形变（按概率）
        if np.random.rand() < self.elastic_prob:
            rs = np.random.RandomState(None)
            dx = gaussian_filter((rs.rand(H, W) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
            dy = gaussian_filter((rs.rand(H, W) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
            x, y = np.meshgrid(np.arange(W), np.arange(H))
            
            if img_affine.ndim == 3:
                img_geo = np.stack([map_coordinates(img_affine[..., ch], [y + dy, x + dx], order=1, mode='reflect') for ch in range(img_affine.shape[-1])], axis=-1)
            else:
                img_geo = map_coordinates(img_affine, [y + dy, x + dx], order=1, mode='reflect')
            if mask_affine.ndim == 3:
                mask_geo = np.stack([map_coordinates(mask_affine[..., ch], [y + dy, x + dx], order=0, mode='nearest') for ch in range(mask_affine.shape[-1])], axis=-1)
            else:
                mask_geo = map_coordinates(mask_affine, [y + dy, x + dx], order=0, mode='nearest')
        else:
            img_geo, mask_geo = img_affine, mask_affine

        # —— 3) 强度类增强 (仅对图像) —— #
        img_intensity = self._apply_intensity_augmentations(img_geo)
        
        # —— 4) 随机擦除 (Cutout) —— #
        img_final, mask_final = self._apply_cutout(img_intensity, mask_geo)

        # --- 类型与范围 ---
        img_final = img_final.astype(img.dtype) # 恢复输入图像的类型
        mask_final = np.rint(mask_final).astype(mask.dtype)
        return img_final, mask_final



class RandomGeoAugmentor:
    """
    随机 仿射(旋转/平移/剪切/缩放) + 弹性形变 的增强器
    参数仅在此处定义，生成器无需知道这些参数
    """
    def __init__(self,
                 max_rotate_deg=15,     # ±15°
                 max_shift_ratio=0.05,  # 宽高的 5%
                 max_zoom_ratio=0.10,   # 0.9~1.1
                 max_shear_deg=10,      # ±10°
                 elastic_alpha=34,      # 弹性强度
                 elastic_sigma=4,       # 弹性平滑
                 elastic_prob=0.5):
        self.max_rotate_deg = max_rotate_deg
        self.max_shift_ratio = max_shift_ratio
        self.max_zoom_ratio = max_zoom_ratio
        self.max_shear_deg = max_shear_deg
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob

    def __call__(self, img, mask):
        """img:(H,W,C), mask:(H,W,1 or Cmask) -> 同步增强后的(img, mask)"""
        H, W = img.shape[:2]

        # —— 1) 仿射参数 —— #
        angle = np.deg2rad(np.random.uniform(-self.max_rotate_deg, self.max_rotate_deg))
        shear = np.deg2rad(np.random.uniform(-self.max_shear_deg, self.max_shear_deg))
        scale = 1.0 + np.random.uniform(-self.max_zoom_ratio, self.max_zoom_ratio)
        tx = np.random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * W
        ty = np.random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * H

        tform = AffineTransform(scale=(scale, scale),
                                rotation=angle,
                                shear=shear,
                                translation=(tx, ty))

        img_affine = warp(img, tform.inverse, order=1, mode='constant',cval = 0, preserve_range=True)
        if mask.ndim == 2:
            mask_affine = warp(mask, tform.inverse, order=0, mode='constant', cval=0, preserve_range=True)
        else:
            mask_affine = np.stack([
                warp(mask[..., c], tform.inverse, order=0, mode='constant', cval=0, preserve_range=True)
                for c in range(mask.shape[-1])
            ], axis=-1)

        # —— 2) 弹性形变（按概率） —— #
        if np.random.rand() < self.elastic_prob:
            rs = np.random.RandomState(None)
            dx = gaussian_filter((rs.rand(H, W) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
            dy = gaussian_filter((rs.rand(H, W) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
            x, y = np.meshgrid(np.arange(W), np.arange(H))

            if img_affine.ndim == 3:
                img_elastic = np.stack([
                    map_coordinates(img_affine[..., ch], [y + dy, x + dx], order=1, mode='reflect')
                    for ch in range(img_affine.shape[-1])
                ], axis=-1)
            else:
                img_elastic = map_coordinates(img_affine, [y + dy, x + dx], order=1, mode='reflect')

            if mask_affine.ndim == 3:
                mask_elastic = np.stack([
                    map_coordinates(mask_affine[..., ch], [y + dy, x + dx], order=0, mode='nearest')
                    for ch in range(mask_affine.shape[-1])
                ], axis=-1)
            else:
                mask_elastic = map_coordinates(mask_affine, [y + dy, x + dx], order=0, mode='nearest')

            img_out, mask_out = img_elastic, mask_elastic
        else:
            img_out, mask_out = img_affine, mask_affine

        # 类型与范围
        img_out = img_out.astype(np.float32)
        mask_out = np.rint(mask_out).astype(np.int32)
        return img_out, mask_out




aug_fn = EnhancedAugmentor()

class CustomDataGenerator(tf.keras.utils.Sequence):
    """
    Keras 自定义数据生成器
    """
    def __init__(self, image_filenames, mask_filenames, batch_size, dim, n_classes,
                 augment = True, shuffle=True):
        '初始化'
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        '返回每个 epoch 的批次数'
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        '生成一个批次的数据'
        # 1. 获取这个批次的索引
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # 2. 根据索引找到对应的文件路径
        batch_image_files = [self.image_filenames[k] for k in indexes]
        batch_mask_files = [self.mask_filenames[k] for k in indexes]

        # 3. 从硬盘加载并预处理数据
        X, y = self.__data_generation(batch_image_files, batch_mask_files)
        
        return X, y

    def on_epoch_end(self):
        '在每个 epoch 结束后，如果需要则打乱数据顺序'
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_files, batch_mask_files):
        '核心函数：加载、预处理、增强'
        X = np.empty((self.batch_size, *self.dim),dtype= np.float32) # self.dim 是 (H, W, C)，例如 (256, 256, 1)
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_classes),dtype= np.float32) # one-hot 编码


        for i, (img_path, mask_path) in enumerate(zip(batch_image_files, batch_mask_files)):
            # --- 加载图像和掩码 ---
            img = imread(img_path) # 读取灰度图
            mask = imread(mask_path)
            img = resize(img, (self.dim[0],self.dim[1]), mode='constant', preserve_range=True)
            mask = resize(mask, (self.dim[0],self.dim[1]), order = 0,mode='constant', preserve_range=True,anti_aliasing=False)

            if img.ndim == 2: img = np.expand_dims(img, axis=-1)  # (H,W) -> (H,W,1)
            if mask.ndim == 2: mask = np.expand_dims(mask, axis=-1)

            if img.max()>0:img/=255
            if self.augment:
                img,mask = aug_fn(img,mask)
            
            img_tf = tf.convert_to_tensor(img, dtype=tf.float32)
            mask_tf = tf.convert_to_tensor(mask, dtype=tf.int32)

            mask_oh = tf.one_hot(tf.squeeze(mask_tf, axis=-1), depth=self.n_classes, dtype=tf.float32)


            X[i,] = img_tf.numpy()
            y[i,] = mask_oh.numpy()
        return X, y