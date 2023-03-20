import os
import sys
import random
import argparse
import numpy as np
import datetime
from PIL import Image

import paddle

parent_path = os.path.abspath(os.path.join(__file__, *(['..']*4)))
sys.path.insert(0, parent_path)
import ppindustry.uad.datasets.mvtec as mvtec
from ppindustry.uad.models.patchcore import get_model
from ppindustry.uad.utils.utils import plot_fig, str2bool
from ppindustry.cvlib.uad_configs import ConfigParser


def argsparser():
    parser = argparse.ArgumentParser('PatchCore')
    parser.add_argument("--config", type=str, default=None, help="Path of config", required=True)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None, help="specify model path if needed")
    parser.add_argument("--category", type=str, default=None, help="category name for MvTec AD dataset")
    parser.add_argument('--resize', type=list or tuple, default=None)
    parser.add_argument('--crop_size', type=list or tuple, default=None)
    parser.add_argument("--backbone", type=str, default=None,
                        help="backbone model arch, one of [resnet18, resnet50, wide_resnet50_2]")
    parser.add_argument("--k", type=int, default=None, help="feature used")
    parser.add_argument("--method", type=str, default=None, help="projection method, one of [sample,ortho]")
    parser.add_argument("--save_pic", type=str2bool, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument("--norm", type=str2bool, default=True)
    return parser.parse_args()


def main():
    args = argsparser()
    config_parser = ConfigParser(args)
    args = config_parser.parser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.device.set_device(args.device)

    class_name = args.category
    assert class_name in mvtec.CLASS_NAMES
    print("Testing model for {}".format(class_name))
    # build model
    model = get_model(args.method)(arch=args.backbone, pretrained=False, k=args.k, method=args.method)
    model.eval()
    state = paddle.load(args.model_path)
    model.model.set_dict(state["params"])
    model.load(state["stats"])
    model.eval()

    # build data
    transform_x = mvtec.MVTecDataset.get_transform(resize=args.resize, cropsize=args.crop_size)[0]
    x = Image.open(args.img_path).convert('RGB')
    x = transform_x(x).unsqueeze(0)
    predict(args, model, x)


def predict(args, model, x):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Starting eval model...")
    # extract test set features
    # model prediction
    out = model(x)
    out = model.project(out)
    score_map, image_score = model.generate_scores_map(out, x.shape[-2:])
    # score_map = np.concatenate(score_map, 0)

    # Normalization
    if args.norm:
        max_score = score_map.max()
        min_score = score_map.min()
        score_map = (score_map - min_score) / (max_score - min_score)
    save_name = os.path.join(args.save_path + f"/{args.method}_{args.backbone}_{args.k}", args.category)
    dir_name = os.path.dirname(save_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plot_fig(x.numpy(), score_map, None, args.threshold, save_name, args.category, args.save_pic, 'predict')

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Predict :  Picture {}".format(
        args.img_path) + " done!")
    if args.save_pic: print("Result saved at {}/{}_predict.png".format(save_name, args.category))


if __name__ == '__main__':
    main()
