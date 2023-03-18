from train_agilenn import (train_agilenn_cifar10, 
                           train_agilenn_cifar100,
                           train_agilenn_svhn,
                           train_agilenn_imagenet200)
from models import MobileNetV2_AgileNN
from train_evaluator import construct_evaluator
import argparse


parser = argparse.ArgumentParser(description='AgileNN training configs')

parser.add_argument('--dataset', type=str, default='cifar100', help='valid datasets are cifar10, cifar100, svhn, imagenet200')
parser.add_argument('--split_ratio', type=float, default=0.2, help='num_local_features / (num_local_features + num_remote_features)')
parser.add_argument('--rho', type=float, default=0.8, help='skewness of feature importance')
parser.add_argument('--klambda', type=float, default=0.8, help='to balance loss terms, lambda * L_ce + (1 - lambda) (L_skewness + L_disorder)')
parser.add_argument('--num_centroids', type=int, default=8, help='quantize remote features to log2(num_centroids) bit representation ')

args = parser.parse_args()
dataset = args.dataset
split_ratio = args.split_ratio
rho = args.rho
klambda = args.klambda
num_centroids = args.num_centroids

evaluator_path = 'saved_models/effnetv2_pretrained' + '_' + dataset + '.tf'

if dataset == 'cifar10':
    evaluator = construct_evaluator(evaluator_path,
                                    feature_size=32,
                                    num_classes=10)
    model = MobileNetV2_AgileNN(classes=10, 
                                data_format='channels_last',
                                conv1_stride=3,
                                split_ratio=split_ratio,
                                num_centroids=num_centroids)
    
    train_agilenn_cifar10(model, 
                          evaluator, 
                          run_name='agilenn_mobilenetv2_cifar10', 
                          logdir='logs',
                          split_ratio=split_ratio,
                          rho=rho,
                          klambda=klambda)
    
elif dataset == 'cifar100':
    evaluator = construct_evaluator(evaluator_path,
                                    feature_size=32,
                                    num_classes=100)
    model = MobileNetV2_AgileNN(classes=100, 
                                data_format='channels_last',
                                conv1_stride=3,
                                split_ratio=split_ratio,
                                num_centroids=num_centroids)
    
    train_agilenn_cifar100(model, 
                           evaluator, 
                           run_name='agilenn_mobilenetv2_cifar100', 
                           logdir='logs',
                           split_ratio=split_ratio,
                           rho=rho,
                           klambda=klambda)
    
elif dataset == 'svhn':
    evaluator = construct_evaluator(evaluator_path,
                                    feature_size=32,
                                    num_classes=10)
    model = MobileNetV2_AgileNN(classes=10, 
                                data_format='channels_last',
                                conv1_stride=3,
                                split_ratio=split_ratio,
                                num_centroids=num_centroids)
    
    train_agilenn_svhn(model, 
                       evaluator, 
                       run_name='agilenn_mobilenetv2_svhn', 
                       logdir='logs',
                       split_ratio=split_ratio,
                       rho=rho,
                       klambda=klambda)
    
elif dataset == 'imagenet200':
    evaluator = construct_evaluator(evaluator_path,
                                    feature_size=64,
                                    num_classes=200)
    model = MobileNetV2_AgileNN(classes=200, 
                                data_format='channels_last',
                                conv1_stride=2,
                                split_ratio=split_ratio,
                                num_centroids=num_centroids)
    
    train_agilenn_imagenet200(model, 
                              evaluator, 
                              run_name='agilenn_mobilenetv2_imagenet200', 
                              logdir='logs',
                              split_ratio=split_ratio,
                              rho=rho,
                              klambda=klambda)

else:
    raise NotImplementedError("This dataset has not been implemented yet")