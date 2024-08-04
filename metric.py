from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torch
import argparse
import tqdm
import os
import cv2
import numpy as np
from einops import rearrange
from src.dataset.utils import get_K_R
from collections import defaultdict

torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_dir', type=str, default="/home/yxy/work/mvdiffusion/logs/tb_logs/val=1/version_0/images",
                        help='Directory where results are saved')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')

    return parser.parse_args()


class MVResultDataset(torch.utils.data.Dataset):
    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.scenes = os.listdir(result_dir)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        num_views = 8
        images_gt = []
        images_gen = []
        Rs = []
        cameras = defaultdict(list)
        for i in range(num_views):
            for images, ext in zip([images_gt, images_gen], ["_natural.png", ".png"]):
                img = cv2.imread(os.path.join(self.result_dir, self.scenes[idx], f"{i}{ext}"))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

            theta = (360 / num_views * i) % 360
            K, R = get_K_R(90, theta, 0, *img.shape[:2])

            Rs.append(R)
            cameras['height'].append(img.shape[0])
            cameras['width'].append(img.shape[1])
            cameras['FoV'].append(90)
            cameras['theta'].append(theta)
            cameras['phi'].append(0)

        images_gt = np.stack(images_gt, axis=0)
        images_gen = np.stack(images_gen, axis=0)
        K = np.stack([K]*len(Rs)).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)
        for k, v in cameras.items():
            cameras[k] = np.stack(v)

        prompt_dir = os.path.join(self.result_dir, self.scenes[idx], "prompt.txt")
        prompt = []
        with open(prompt_dir, 'r') as f:
            for line in f:
                prompt.append(line.strip())

        return {
            'images_gt': images_gt,
            'images_gen': images_gen,
            'K': K,
            'R': R,
            'cameras': cameras,
            'prompt': prompt
        }


if __name__ == '__main__':
    args = parse_args()

    fid = FrechetInceptionDistance(feature=2048).cuda()
    kid = KernelInceptionDistance(feature=2048).cuda()
    inception = InceptionScore().cuda()

    dataset = MVResultDataset(args.result_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size)

    for batch in tqdm.tqdm(data_loader):
        images_gt = rearrange(batch['images_gt'].cuda(), 'b l h w c -> (b l) c h w')
        images_gen = rearrange(batch['images_gen'].cuda(), 'b l h w c -> (b l) c h w')
        fid.update(images_gt, real=True)
        fid.update(images_gen, real=False)
        kid.update(images_gt, real=True)
        kid.update(images_gen, real=False)
        inception.update(images_gen)

        prompt_reshape = sum(map(list, zip(*batch['prompt'])), [])

    print(f"FID: {fid.compute()}")
    print(f"KID: {kid.compute()}")
    print(f"IS: {inception.compute()}")