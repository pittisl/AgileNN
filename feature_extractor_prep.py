import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from train_evaluator import construct_evaluator
from models import MobileNetV2_AgileNN
from tiny_imagenet import TinyImagenetDataset


def channel_remapping_cifar10(
    model, 
    evaluator, 
    split_ratio, 
    metric='global_importance'
):
    K_top = int(np.round(split_ratio * 24))
    
    ds = tfds.load('cifar10', as_supervised=True)
    
    std = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
    mean= tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))
    
    def train_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.random_flip_left_right(x)
        x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - mean) / std
        x = tf.image.resize(x, [96, 96])
        return x, y
    
    ds_train = ds['train'].map(train_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                    .shuffle(1024)\
                                    .batch(128)\
                                    .prefetch(buffer_size=tf.data.AUTOTUNE)
                                    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    sort_metric = tf.metrics.MeanTensor()
    
    if metric == 'global_importance':
        
        @tf.function
        def step(x, y, training):
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2]) # [batch, channel]
            
            sort_metric(tf.reduce_mean(per_channel_importance, axis=0))
                
    elif metric == 'topk_appearance':
        
        @tf.function
        def step(x, y, training):
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2]) # [batch, channel]
            sorted_importance = tf.sort(per_channel_importance, axis=-1, direction='DESCENDING') # [batch, channel]
            kth_importance = sorted_importance[K_top] # grab the kth high importance value
            does_belong_to_topk = tf.cast(sorted_importance >= kth_importance, tf.float32) # whether each channel belongs to top-k
            
            sort_metric(tf.reduce_mean(does_belong_to_topk, axis=0))
    
    else:
        raise NotImplementedError("This metric has not been implemented yet")
    
    # traverse entire training set and accumulate the `sort_metric`
    for x, y in tqdm(ds_train, ascii=True):
        step(x, y, training=False)
        
    # based on the accmulated sort_metric, sort output channels of feature extractor by exchanging fitlers' weights
    feature_extractor_weights = model.feature_extractor_1.trainable_weights
    filters_weights_src = feature_extractor_weights[5].value().numpy() # refers to `separable_conv2d_1/pointwise_kernel:0` shape=[1,1,4,24]
    filters_weights_des = feature_extractor_weights[5].value().numpy()
    # obtain new indices
    scores = sort_metric.result() # (24,)
    mapping = tf.argsort(scores, direction='DESCENDING')
    # ground
    print('Original channels are mapped to new positions:')
    for pos_des, pos_src in enumerate(mapping):
        filters_weights_des[:, :, :, pos_des] = filters_weights_src[:, :, :, pos_src]
        print(f'{pos_src} -> {pos_des}')
    model.feature_extractor_1.trainable_weights[5].assign(filters_weights_des)
    
    return model


def channel_remapping_cifar100(
    model, 
    evaluator, 
    split_ratio, 
    metric='global_importance'
):
    K_top = int(np.round(split_ratio * 24))
    
    ds = tfds.load('cifar100', as_supervised=True)
    
    std = tf.reshape((0.267, 0.256, 0.276), shape=(1, 1, 3))
    mean= tf.reshape((0.507, 0.487, 0.441), shape=(1, 1, 3))
    
    def train_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.random_flip_left_right(x)
        x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - mean) / std
        x = tf.image.resize(x, [96, 96])
        return x, y
    
    ds_train = ds['train'].map(train_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                    .shuffle(1024)\
                                    .batch(64)\
                                    .prefetch(buffer_size=tf.data.AUTOTUNE)
                                    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    sort_metric = tf.metrics.MeanTensor()
    
    if metric == 'global_importance':
        
        @tf.function
        def step(x, y, training):
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2]) # [batch, channel]
            
            sort_metric(tf.reduce_mean(per_channel_importance, axis=0))
                
    elif metric == 'topk_appearance':
        
        @tf.function
        def step(x, y, training):
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2]) # [batch, channel]
            sorted_importance = tf.sort(per_channel_importance, axis=-1, direction='DESCENDING') # [batch, channel]
            kth_importance = sorted_importance[K_top] # grab the kth high importance value
            does_belong_to_topk = tf.cast(sorted_importance >= kth_importance, tf.float32) # whether each channel belongs to top-k
            
            sort_metric(tf.reduce_mean(does_belong_to_topk, axis=0))
    
    else:
        raise NotImplementedError("This metric has not been implemented yet")
    
    # traverse entire training set and accumulate the `sort_metric`
    for x, y in tqdm(ds_train, ascii=True):
        step(x, y, training=False)
        
    # based on the accmulated sort_metric, sort output channels of feature extractor by exchanging fitlers' weights
    feature_extractor_weights = model.feature_extractor_1.trainable_weights
    filters_weights_src = feature_extractor_weights[5].value().numpy() # refers to `separable_conv2d_1/pointwise_kernel:0` shape=[1,1,4,24]
    filters_weights_des = feature_extractor_weights[5].value().numpy()
    # obtain new indices
    scores = sort_metric.result() # (24,)
    mapping = tf.argsort(scores, direction='DESCENDING')
    # ground
    print('Original channels are mapped to new positions:')
    for pos_des, pos_src in enumerate(mapping):
        filters_weights_des[:, :, :, pos_des] = filters_weights_src[:, :, :, pos_src]
        print(f'{pos_src} -> {pos_des}')
    model.feature_extractor_1.trainable_weights[5].assign(filters_weights_des)
    
    return model


def channel_remapping_svhn(
    model, 
    evaluator, 
    split_ratio, 
    metric='global_importance'
):
    K_top = int(np.round(split_ratio * 24))
    
    ds = tfds.load('svhn_cropped', as_supervised=True)
    
    def train_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.random_flip_left_right(x)
        x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = 2 * x - 1
        x = tf.image.resize(x, [96, 96])
        return x, y
    
    ds_train = ds['train'].map(train_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                    .shuffle(1024)\
                                    .batch(64)\
                                    .prefetch(buffer_size=tf.data.AUTOTUNE)
                                    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    sort_metric = tf.metrics.MeanTensor()
    
    if metric == 'global_importance':
        
        @tf.function
        def step(x, y, training):
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2]) # [batch, channel]
            
            sort_metric(tf.reduce_mean(per_channel_importance, axis=0))
                
    elif metric == 'topk_appearance':
        
        @tf.function
        def step(x, y, training):
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2]) # [batch, channel]
            sorted_importance = tf.sort(per_channel_importance, axis=-1, direction='DESCENDING') # [batch, channel]
            kth_importance = sorted_importance[K_top] # grab the kth high importance value
            does_belong_to_topk = tf.cast(sorted_importance >= kth_importance, tf.float32) # whether each channel belongs to top-k
            
            sort_metric(tf.reduce_mean(does_belong_to_topk, axis=0))
    
    else:
        raise NotImplementedError("This metric has not been implemented yet")
    
    # traverse entire training set and accumulate the `sort_metric`
    for x, y in tqdm(ds_train, ascii=True):
        step(x, y, training=False)
        
    # based on the accmulated sort_metric, sort output channels of feature extractor by exchanging fitlers' weights
    feature_extractor_weights = model.feature_extractor_1.trainable_weights
    filters_weights_src = feature_extractor_weights[5].value().numpy() # refers to `separable_conv2d_1/pointwise_kernel:0` shape=[1,1,4,24]
    filters_weights_des = feature_extractor_weights[5].value().numpy()
    # obtain new indices
    scores = sort_metric.result() # (24,)
    mapping = tf.argsort(scores, direction='DESCENDING')
    # ground
    print('Original channels are mapped to new positions:')
    for pos_des, pos_src in enumerate(mapping):
        filters_weights_des[:, :, :, pos_des] = filters_weights_src[:, :, :, pos_src]
        print(f'{pos_src} -> {pos_des}')
    model.feature_extractor_1.trainable_weights[5].assign(filters_weights_des)
    
    return model


def channel_remapping_imagenet200(
    model, 
    evaluator, 
    split_ratio, 
    metric='global_importance'
):
    K_top = int(np.round(split_ratio * 24))
    
    tiny_imagenet_builder = TinyImagenetDataset()
    tiny_imagenet_builder.download_and_prepare()
    train_dataset = tiny_imagenet_builder.as_dataset(split="train")
    
    def train_prep(sample):
        x, y = sample["image"], sample["label"]
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.random_flip_left_right(x)
        x = 2 * x - 1
        x = tf.image.resize(x, [128, 128])
        return x, y
    
    ds_train = train_dataset.map(train_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                 .shuffle(1024)\
                                 .batch(64)\
                                 .prefetch(buffer_size=tf.data.AUTOTUNE)
                                    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    sort_metric = tf.metrics.MeanTensor()
    
    if metric == 'global_importance':
        
        @tf.function
        def step(x, y, training):
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2]) # [batch, channel]
            
            sort_metric(tf.reduce_mean(per_channel_importance, axis=0))
                
    elif metric == 'topk_appearance':
        
        @tf.function
        def step(x, y, training):
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2]) # [batch, channel]
            sorted_importance = tf.sort(per_channel_importance, axis=-1, direction='DESCENDING') # [batch, channel]
            kth_importance = sorted_importance[K_top] # grab the kth high importance value
            does_belong_to_topk = tf.cast(sorted_importance >= kth_importance, tf.float32) # whether each channel belongs to top-k
            
            sort_metric(tf.reduce_mean(does_belong_to_topk, axis=0))
    
    else:
        raise NotImplementedError("This metric has not been implemented yet")
    
    # traverse entire training set and accumulate the `sort_metric`
    for x, y in tqdm(ds_train, ascii=True):
        step(x, y, training=False)
        
    # based on the accmulated sort_metric, sort output channels of feature extractor by exchanging fitlers' weights
    feature_extractor_weights = model.feature_extractor_1.trainable_weights
    filters_weights_src = feature_extractor_weights[5].value().numpy() # refers to `separable_conv2d_1/pointwise_kernel:0` shape=[1,1,4,24]
    filters_weights_des = feature_extractor_weights[5].value().numpy()
    # obtain new indices
    scores = sort_metric.result() # (24,)
    mapping = tf.argsort(scores, direction='DESCENDING')
    # ground
    print('Original channels are mapped to new positions:')
    for pos_des, pos_src in enumerate(mapping):
        filters_weights_des[:, :, :, pos_des] = filters_weights_src[:, :, :, pos_src]
        print(f'{pos_src} -> {pos_des}')
    model.feature_extractor_1.trainable_weights[5].assign(filters_weights_des)
    
    return model


def main():
    evaluator_path = 'saved_models/effnetv2_pretrained' + '_' + 'cifar100' + '.tf'
    evaluator = construct_evaluator(evaluator_path,
                                    feature_size=32,
                                    num_classes=100)
    model = MobileNetV2_AgileNN(classes=100, 
                                data_format='channels_last',
                                conv1_stride=3,
                                split_ratio=0.2,
                                num_centroids=8)
    channel_remapping_cifar100(model, evaluator, split_ratio=0.2, metric='topk_appearance')

if __name__ == '__main__':
    main()
    
    