import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import argparse
from tqdm import tqdm
from tiny_imagenet import TinyImagenetDataset # https://github.com/ksachdeva/tiny-imagenet-tfds


def construct_effnetv2(image_size=224, num_classes=100):
    effnetv2 = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3),
        include_preprocessing=False) # expected pixel value range [-1, 1]
    model = tf.keras.Sequential()
    model.add(effnetv2)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes))
    return model


def train_effnetv2_on_cifar10(run_name, logdir):
    model = construct_effnetv2(num_classes=10)
    
    ds = tfds.load('cifar10', as_supervised=True)
    
    def prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = 2 * x - 1
        x = tf.image.resize(x, [224, 224])
        return x, y
    
    ds_train = ds['train'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                 .shuffle(1024)\
                                 .batch(8)\
                                 .prefetch(buffer_size=tf.data.AUTOTUNE)
                                 
    ds_test = ds['test'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(8*4)\
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
    
    lr = 1e-4
    epochs = 15
    decay_steps = len(tfds.as_numpy(ds_train)) * epochs
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps=decay_steps, alpha=1e-2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    runid = run_name + '_cifar10' # '_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    
    print(f"RUNID: {runid}")
    
    @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=training)
            ce_loss = loss_fn_cls(y, y_pred)
            loss = ce_loss
        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))     

        accuracy(y, y_pred)
        cls_loss(ce_loss)

    training_step = 0
    best_validation_acc = 0

    for epoch in range(epochs):
        
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1
            
            step(x, y, training=True)
            
            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
        
        
        cls_loss.reset_states()
        accuracy.reset_states()

        for x, y in ds_test:
            step(x, y, training=False)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)
            
            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")
                
            cls_loss.reset_states()
            accuracy.reset_states()


def train_effnetv2_on_cifar100(run_name, logdir):
    model = construct_effnetv2(num_classes=100)
    
    ds = tfds.load('cifar100', as_supervised=True)
    
    def prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = 2 * x - 1
        x = tf.image.resize(x, [224, 224])
        return x, y
    
    ds_train = ds['train'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                 .shuffle(1024)\
                                 .batch(8)\
                                 .prefetch(buffer_size=tf.data.AUTOTUNE)
                                 
    ds_test = ds['test'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(8*4)\
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
    
    lr = 1e-4
    epochs = 15
    decay_steps = len(tfds.as_numpy(ds_train)) * epochs
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps=decay_steps, alpha=1e-2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    runid = run_name + '_cifar100' # '_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    
    print(f"RUNID: {runid}")
    
    @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=training)
            ce_loss = loss_fn_cls(y, y_pred)
            loss = ce_loss
        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))     

        accuracy(y, y_pred)
        cls_loss(ce_loss)

    training_step = 0
    best_validation_acc = 0

    for epoch in range(epochs):
        
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1
            
            step(x, y, training=True)
            
            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
        
        
        cls_loss.reset_states()
        accuracy.reset_states()

        for x, y in ds_test:
            step(x, y, training=False)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)
            
            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")
                
            cls_loss.reset_states()
            accuracy.reset_states()


def train_effnetv2_on_svhn(run_name, logdir):
    model = construct_effnetv2(num_classes=10)
    
    ds = tfds.load('svhn_cropped', as_supervised=True)
    
    def prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = 2 * x - 1
        x = tf.image.resize(x, [224, 224])
        return x, y
    
    ds_train = ds['train'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                 .shuffle(1024)\
                                 .batch(8)\
                                 .prefetch(buffer_size=tf.data.AUTOTUNE)
                                 
    ds_test = ds['test'].map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(8*4)\
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
    
    lr = 1e-4
    epochs = 15
    decay_steps = len(tfds.as_numpy(ds_train)) * epochs
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps=decay_steps, alpha=1e-2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    runid = run_name + '_svhn' # '_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    
    print(f"RUNID: {runid}")
    
    @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=training)
            ce_loss = loss_fn_cls(y, y_pred)
            loss = ce_loss
        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))     

        accuracy(y, y_pred)
        cls_loss(ce_loss)

    training_step = 0
    best_validation_acc = 0

    for epoch in range(epochs):
        
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1
            
            step(x, y, training=True)
            
            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
        
        
        cls_loss.reset_states()
        accuracy.reset_states()

        for x, y in ds_test:
            step(x, y, training=False)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)
            
            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")
                
            cls_loss.reset_states()
            accuracy.reset_states()


def train_effnetv2_on_imagenet200(run_name, logdir):
    model = construct_effnetv2(num_classes=200)
    
    tiny_imagenet_builder = TinyImagenetDataset()
    tiny_imagenet_builder.download_and_prepare()
    train_dataset = tiny_imagenet_builder.as_dataset(split="train")
    test_dataset = tiny_imagenet_builder.as_dataset(split="validation")
    

    def prep(sample):
        x, y = sample["image"], sample["label"]
        x = tf.cast(x, tf.float32)/255.
        x = 2 * x - 1
        x = tf.image.resize(x, [224, 224])
        return x, y
    
    ds_train = train_dataset.map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                 .shuffle(1024)\
                                 .batch(8)\
                                 .prefetch(buffer_size=tf.data.AUTOTUNE)
                                 
    ds_test = test_dataset.map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(8*4)\
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
    
    lr = 1e-4
    epochs = 15
    decay_steps = len(tfds.as_numpy(ds_train)) * epochs
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps=decay_steps, alpha=1e-2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    runid = run_name + '_imagenet200' # '_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    
    print(f"RUNID: {runid}")
    
    @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=training)
            ce_loss = loss_fn_cls(y, y_pred)
            loss = ce_loss
        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))     

        accuracy(y, y_pred)
        cls_loss(ce_loss)

    training_step = 0
    best_validation_acc = 0

    for epoch in range(epochs):
        
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1
            
            step(x, y, training=True)
            
            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
        
        
        cls_loss.reset_states()
        accuracy.reset_states()

        for x, y in ds_test:
            step(x, y, training=False)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)
            
            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))
                print("=================================")
                print("acc: ", accuracy.result())
                print("=================================")
                
            cls_loss.reset_states()
            accuracy.reset_states()

        
def construct_evaluator(model_path, feature_size=32, num_classes=100):
    effnetv2 = construct_effnetv2(image_size=feature_size, num_classes=num_classes)
    effnetv2.load_weights(model_path)
    effnetv2_backbone = effnetv2.layers[0]
    config = effnetv2_backbone.get_config()
    
    config['layers'][4]['inbound_nodes'] = config['layers'][1]['inbound_nodes']
    config['layers'][0]['config']['batch_input_shape'] = (None, feature_size, feature_size, 24) 
    config['layers'][1:4] = [] # delete first conv layer 
    config['layers'][4]['inbound_nodes'][0][1] = config['layers'][1]['inbound_nodes'][0][0]
    
    headless_backbone = tf.keras.Model.from_config(config)
    
    model = tf.keras.Sequential()
    model.add(headless_backbone)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes))
    
    for v1, v2 in zip(model.variables, effnetv2.variables[5:]):
        v1.assign(v2.value())
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reference NN training configs')
    parser.add_argument('--dataset', type=str, default='cifar100', help='valid datasets are cifar10, cifar100, svhn, imagenet200')
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'cifar10':
        train_effnetv2_on_cifar10('effnetv2_pretrained', 'logs')
    elif dataset == 'cifar100':
        train_effnetv2_on_cifar100('effnetv2_pretrained', 'logs')
    elif dataset == 'svhn':
        train_effnetv2_on_svhn('effnetv2_pretrained', 'logs')
    elif dataset == 'imagenet200':
        train_effnetv2_on_imagenet200('effnetv2_pretrained', 'logs')
    else:
        raise NotImplementedError("This dataset has not been implemented yet")
    
    