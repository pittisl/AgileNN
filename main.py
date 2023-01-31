from train_agilenn import (train_agilenn_cifar10, 
                           train_agilenn_cifar100,
                           train_agilenn_svhn,
                           train_agilenn_imagenet200)
from models import MobileNetV2_AgileNN
from train_evaluator import construct_evaluator


# training config of AgileNN
DATASET = 'cifar100' # dataset to be trained on, selected from ['cifar10', 'cifar100', 'svhn', 'imagenet200']
EVALUATOR_PATH = 'saved_models/effnetv2_pretrained_x1886.tf'
SPLIT_RATIO = 0.2 # num_local_features / (num_local_features + num_remote_features)
RHO = 0.8 # skewness of feature importance
LAMBDA = 0.8 # to balance loss terms, lambda * L_ce + (1 - lambda) (L_skewness + L_disorder)
NUM_CENTROIDS = 8 # quantize remote features to log2(NUM_CENTROIDS) bit representation 


if DATASET == 'cifar10':
    evaluator = construct_evaluator(EVALUATOR_PATH,
                                    feature_size=32,
                                    num_classes=10)
    model = MobileNetV2_AgileNN(classes=10, 
                                data_format='channels_last',
                                conv1_stride=3,
                                split_ratio=SPLIT_RATIO,
                                num_centroids=NUM_CENTROIDS)
    
    train_agilenn_cifar10(model, 
                          evaluator, 
                          run_name='agilenn_mobilenetv2_cifar10', 
                          logdir='logs',
                          split_ratio=SPLIT_RATIO,
                          rho=RHO,
                          klambda=LAMBDA)
    
elif DATASET == 'cifar100':
    evaluator = construct_evaluator(EVALUATOR_PATH,
                                    feature_size=32,
                                    num_classes=100)
    model = MobileNetV2_AgileNN(classes=100, 
                                data_format='channels_last',
                                conv1_stride=3,
                                split_ratio=SPLIT_RATIO,
                                num_centroids=NUM_CENTROIDS)
    
    train_agilenn_cifar100(model, 
                           evaluator, 
                           run_name='agilenn_mobilenetv2_cifar100', 
                           logdir='logs',
                           split_ratio=SPLIT_RATIO,
                           rho=RHO,
                           klambda=LAMBDA)
    
elif DATASET == 'svhn':
    evaluator = construct_evaluator(EVALUATOR_PATH,
                                    feature_size=32,
                                    num_classes=10)
    model = MobileNetV2_AgileNN(classes=10, 
                                data_format='channels_last',
                                conv1_stride=3,
                                split_ratio=SPLIT_RATIO,
                                num_centroids=NUM_CENTROIDS)
    
    train_agilenn_svhn(model, 
                       evaluator, 
                       run_name='agilenn_mobilenetv2_svhn', 
                       logdir='logs',
                       split_ratio=SPLIT_RATIO,
                       rho=RHO,
                       klambda=LAMBDA)
    
elif DATASET == 'imagenet200':
    evaluator = construct_evaluator(EVALUATOR_PATH,
                                    feature_size=64,
                                    num_classes=200)
    model = MobileNetV2_AgileNN(classes=200, 
                                data_format='channels_last',
                                conv1_stride=2,
                                split_ratio=SPLIT_RATIO,
                                num_centroids=NUM_CENTROIDS)
    
    train_agilenn_imagenet200(model, 
                              evaluator, 
                              run_name='agilenn_mobilenetv2_imagenet200', 
                              logdir='logs',
                              split_ratio=SPLIT_RATIO,
                              rho=RHO,
                              klambda=LAMBDA)