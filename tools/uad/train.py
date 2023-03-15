import os
import sys
import time
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from random import sample
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter

import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader

parent_path = os.path.abspath(os.path.join(__file__, *(['..']*3)))
sys.path.insert(0, parent_path)
import ppindustry.uad.datasets.mvtec as mvtec
from ppindustry.uad.models.resnet import ResNet_PaDiM
from ppindustry.cvlib.uad_configs import *


CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

def argsparser():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("--config", type=str, default=None, help="Path of config", required=True)
    parser.add_argument("--device", type=str, default='gpu')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--do_val", type=bool, default=True)
    parser.add_argument("--pretrained_backbone", type=bool, default=True)
    parser.add_argument("--category", type=str, default='zipper')
    parser.add_argument("--data_path", type=str, default='/ssd1/zhaoyantao/PP-Industry/data/mvtec_anomaly_detection')
    parser.add_argument("--save_path", type=str, default='/ssd1/zhaoyantao/PP-Industry/output')
    parser.add_argument("--backbone_depth", type=int, default=18, help="resnet depth")
    parser.add_argument("--print_freq", type=int, default=20)
    return parser.parse_args()


def main():
    args = argsparser()
    config_parser = ConfigParser(args)
    args = config_parser.parser()

    random.seed(args.seed)
    paddle.seed(args.seed)

    # build model
    model = ResNet_PaDiM(depth=args.backbone_depth, pretrained=args.pretrained_backbone).to(args.device)
    model.eval()

    t_d, d = 448, 100 # "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4},
    class_name = args.category
    assert class_name in mvtec.CLASS_NAMES
    print("Training model for {}".format(class_name))

    # build datasets
    train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    if args.do_val:
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    idx = paddle.to_tensor(sample(range(0, t_d), d))

    train(args, model, train_dataloader, idx)

    if args.do_val:
        val(model, test_dataloader, class_name, idx)


def train(args, model, train_dataloader, idx):
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # extract train set features
    epoch_begin = time.time()
    end_time = time.time()

    for index, item in enumerate(train_dataloader):
        start_time = time.time()
        data_time = start_time - end_time
        x = item[0]

        # model prediction
        with paddle.no_grad():
            outputs = model(x)

        # get intermediate layer outputs
        for k, v in zip(train_outputs.keys(), outputs):
            train_outputs[k].append(v.cpu().detach())

        end_time = time.time()
        batch_time = end_time - start_time

        if index % args.print_freq == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
                  "Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, batch time:{:.4f}, data time:{:.4f}".format(
                      0,
                      index + 1,
                      len(train_dataloader),
                      0,
                      float(0),
                      float(batch_time),
                      float(data_time)
                  ))

    for k, v in train_outputs.items():
        train_outputs[k] = paddle.concat(v, 0)

    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        layer_embedding = train_outputs[layer_name]
        layer_embedding = F.interpolate(layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
        embedding_vectors = paddle.concat((embedding_vectors, layer_embedding), 1)

    # randomly select d dimension
    embedding_vectors = paddle.index_select(embedding_vectors,  idx, 1)
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape((B, C, H * W))
    mean = paddle.mean(embedding_vectors, axis=0).numpy()
    cov = paddle.zeros((C, C, H * W)).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    # save learned distribution
    train_outputs = [mean, cov]
    model.distribution = train_outputs
    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(0, t))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Saving model...")
    save_name = os.path.join(args.save_path, args.category, 'best.pdparams')
    dir_name = os.path.dirname(save_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    state_dict = {
        "params":model.model.state_dict(),
        "distribution":model.distribution,
    }
    paddle.save(state_dict, save_name)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Save model in {}".format(str(save_name)))


def val(model, test_dataloader, class_name, idx):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Starting eval model...")
    total_roc_auc = []
    total_pixel_roc_auc = []

    gt_list = []
    gt_mask_list = []
    test_imgs = []

    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # extract test set features
    for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):

        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        with paddle.no_grad():
            outputs = model(x)
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Eval model...")
    for k, v in test_outputs.items():
        test_outputs[k] = paddle.concat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        layer_embedding = test_outputs[layer_name]
        layer_embedding = F.interpolate(layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
        embedding_vectors = paddle.concat((embedding_vectors, layer_embedding), 1)

    # randomly select d dimension
    embedding_vectors = paddle.index_select(embedding_vectors,  idx, 1)

    # calculate distance matrix
    if paddle.is_compiled_with_cuda():
        def mahalanobis_pd(sample, mean, conv_inv):
            return paddle.sqrt(paddle.matmul(paddle.matmul((sample - mean).t(), conv_inv), (sample - mean)))[0]
        B, C, H, W = embedding_vectors.shape
        embedding_vectors = embedding_vectors.reshape((B, C, H * W)).cuda()
        model.distribution[0] = paddle.to_tensor(model.distribution[0]).cuda()
        model.distribution[1] = paddle.to_tensor(model.distribution[1]).cuda()
        dist_list = []
        for i in range(H * W):
            mean = model.distribution[0][:, i]
            conv_inv = paddle.linalg.inv(model.distribution[1][:, :, i])
            dist = [mahalanobis_pd(sample[:, i], mean, conv_inv).numpy()[0] for sample in embedding_vectors]
            dist_list.append(dist)
    else:
        # calculate distance matrix
        B, C, H, W = embedding_vectors.shape
        embedding_vectors = embedding_vectors.reshape((B, C, H * W)).numpy()
        dist_list = []
        for i in range(H * W):
            mean = model.distribution[0][:, i]
            conv_inv = np.linalg.inv(model.distribution[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)
    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)


    # upsample
    dist_list = paddle.to_tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=x.shape[2:], mode='bilinear',
                              align_corners=False).squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    total_roc_auc.append(img_roc_auc)

    gt_mask = np.asarray(gt_mask_list, dtype=np.int64)

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    total_pixel_roc_auc.append(per_pixel_rocauc)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          'Class:{}'.format(class_name) +':\t'+ 'Image AUC: %.3f' % np.mean(total_roc_auc))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          'Class:{}'.format(class_name) +':\t'+ 'Pixel AUC: %.3f' % np.mean(total_pixel_roc_auc))


if __name__ == '__main__':
    main()
