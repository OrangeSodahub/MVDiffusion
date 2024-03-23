import torch
import os
import cv2
import random
import numpy as np
import copy
from pathlib import Path
from src.dataset.utils import load_from_jsonl


class Hypersimdataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train'):
        self.mode=mode
        super().__init__()
        self.data_list = []
        self.scene_kfs = dict()
        self.valid_ids = dict()
        self.num_views = config['num_views']
        jsonl_path = os.path.join(config['root_dir'], f'{mode}.jsonl')

        self.infos = load_from_jsonl(Path(jsonl_path))
        scenes = list(map(lambda info: info['scene_id'], self.infos))
        print('Loaded {} scenes'.format(len(scenes)))

        for scene_id in sorted(scenes):
            scene_info = list(filter(lambda info: info['scene_id'] == scene_id, self.infos))
            valid_id = list(map(lambda info: info['id'], scene_info))
            self.valid_ids[scene_id] = valid_id

            if config['data_load_mode']=='fix_interval':
                raise NotImplementedError
                max_id = max(valid_id)
                kf_ids=[i for i in range(0, max_id, config['test_interval']) if i in valid_id]
                if len(kf_ids) > 150: # relex this number if using GPU with large memory
                    continue
            else:
                num_kf_ids = int(len(valid_id) * 0.6)
                kf_ids = random.sample(list(range(len(valid_id))), num_kf_ids)

            if len(kf_ids) < self.num_views:
                continue
            
            for i, kf_id in enumerate(kf_ids):
                if mode == 'train' or i == 0 or (config['data_load_mode'] == 'fix_frame' and i % config['num_views'] == 0):
                    self.data_list.append((scene_id, kf_id))

            self.scene_kfs[scene_id] = kf_ids
        
        # number of views
        self.num_views = config['num_views']
        self.resolution = (config['resolution_w'], config['resolution_h'])
        self.data_load_mode=config['data_load_mode']
        self.gen_data_ratio=config['gen_data_ratio']
        print(f"Length of data: {len(self)}.")

    def __len__(self):
        return len(self.data_list)

    def _get_consecutive_kfs_inp(self, scene_id):
        img_key_paths=self.scene_kfs[scene_id]
        img_paths=[]
        masks=[]
        valid_ids=self.valid_ids[scene_id]
        for i in range(len(img_key_paths)-1):
            idx1=int(img_key_paths[i].split('/')[-1].split('.')[0])
            idx2=int(img_key_paths[i+1].split('/')[-1].split('.')[0])
            interval=20
            j=1
            img_paths.append(img_key_paths[i])
            masks.append(True)
            while j*interval+idx1<idx2:
                idx=idx1+j*interval
                if idx in valid_ids:
                    img_paths.append(os.path.join(
                            '/'.join(img_key_paths[0].split('/')[:-1]), str(idx)+'.jpg'))
                    masks.append(False)
                j+=1
        img_paths.append(img_key_paths[-1])
        masks.append(True)

        return img_paths, np.array(masks)

    def _get_consecutive_kfs_fix_frame(self, scene_id, sample_idx):
        scene_kfs = self.scene_kfs[scene_id]
        num_views = self.num_views
        num_kfs = len(scene_kfs)
        idx_base = scene_kfs.index(sample_idx)

        if idx_base + num_views < num_kfs:
            idx_list = np.arange(idx_base, idx_base + num_views)
            kf_ids = [scene_kfs[item] for item in idx_list]
        else:
            num_back = num_kfs - idx_base - 1
            num_front = num_views - 1 - num_back
            idx_start = idx_base - num_front
            idx_list = np.arange(idx_start, idx_start + num_views)
            kf_ids = [scene_kfs[item] for item in idx_list]

        return kf_ids
    
    def _get_consecutive_kfs_inp(self, scene_id, img_path):
        valid_ids=self.valid_ids[scene_id]
        scene_kfs=self.scene_kfs[scene_id]
        num_kfs = len(scene_kfs)
        idx_base = scene_kfs.index(img_path)
        num_views=2
        if idx_base + num_views >= num_kfs:
            idx_base=num_kfs-num_views-1
        idx_list = [idx_base, idx_base + num_views]
        img_key1=scene_kfs[idx_list[0]]
        img_key2=scene_kfs[idx_list[1]]
        img_paths = [img_key1]

        idx1=int(img_key1.split('/')[-1].split('.')[0])
        idx2=int(img_key2.split('/')[-1].split('.')[0])

        _idx_list=list(range(idx1+1, idx2-1))
        random.shuffle(_idx_list)
        
        for idx in _idx_list:
            if idx in valid_ids:
                img_paths.append(os.path.join(
                    '/'.join(img_path.split('/')[:-1]), str(idx)+'.jpg'))
            if len(img_paths)==self.num_views-1:
                break
        if len(img_paths)!=self.num_views-1:
            img_paths=img_paths+[img_paths[-1]]*(self.num_views-len(img_paths)-1)
        img_paths.append(img_key2)
        mask=np.zeros(len(img_paths)).astype(np.bool)
        mask[0]=True
        mask[-1]=True

        return img_paths, mask

    def load_seq(self, scene_id, sample_idxs, resolution):
        scene_info = list(filter(lambda info: info['scene_id'] == scene_id, self.infos))
        images_ori = [cv2.cvtColor(cv2.imread(scene_info[i])[..., :3], cv2.COLOR_BGR2RGB)
                      for i in sample_idxs]

        # load pose
        poses = []
        prompts = []
        depths = []
        for idx in sample_idxs:
            info = np.load(scene_info[idx]["source"])
            pose = info["extrin"]
            poses.append(pose)
            if np.isnan(np.linalg.inv(pose)).any():
                return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
            prompts.append(info["caption"])
            depths.append(info["depths"])

        poses = np.stack(poses, axis=0)  # [num_views, 4, 4]

        # load k
        k = np.load(scene_info[0]["source"])["intrin"]
        images = np.stack([cv2.resize(x, resolution) for x in images_ori]) / 127.5 - 1

        # load depth
        h_ori, w_ori = depths[0].shape
        scale = h_ori / resolution[1]

        k[:2] /= scale

        depths = [cv2.resize(x, resolution,
                             interpolation=cv2.INTER_NEAREST) for x in depths]
        depths = np.stack(depths, axis=0)

        depth_valid_mask = depths > 0
        depth_inv = 1. / (depths + 1e-6)
        depth_max = [depth_inv[i][depth_valid_mask[i]].max()
                     for i in range(depth_inv.shape[0])]
        depth_min = [depth_inv[i][depth_valid_mask[i]].min()
                     for i in range(depth_inv.shape[0])]
        depth_max = np.stack(depth_max, axis=0)[:, None, None]
        depth_min = np.stack(depth_min, axis=0)[
            :, None, None]  # [num_views, 1, 1]
        depth_inv_norm_full = (depth_inv - depth_min) / \
            (depth_max - depth_min + 1e-6) * 2 - 1  # [-1, 1]
        depth_inv_norm_full[~depth_valid_mask] = -2
        depth_inv_norm_full = depth_inv_norm_full.astype(np.float32)
        return images, depths, depth_inv_norm_full, poses, k, prompts

    def __getitem__(self, idx):
        scene_id, sample_idx = self.data_list[idx]
        
        if self.data_load_mode == 'fix_interval':
            image_paths = self.scene_kfs[scene_id]
            mask = np.ones(len(image_paths)).astype(np.bool)
        elif self.data_load_mode == 'fix_frame':
            # p_rand = random.random()
            p_rand = 0
            if self.mode == 'train' and p_rand > self.gen_data_ratio:
                image_paths, mask = self._get_consecutive_kfs_inp(
                    scene_id, img_path)
            else:
                sample_idxs = self._get_consecutive_kfs_fix_frame(
                    scene_id, sample_idx)
                mask = np.ones(len(image_paths)).astype(np.bool)
        elif self.data_load_mode=='two_stage':
            image_paths, mask=self._get_consecutive_kfs_inp(
                scene_id, img_path)

        images, depths, depth_inv_norm, poses, K, prompts = self.load_seq(
            scene_id, sample_idxs, self.resolution)
        
        depth_inv_norm_small= np.stack([cv2.resize(depth_inv_norm[i], (
            self.resolution[0]//8, self.resolution[1]//8), interpolation=cv2.INTER_NEAREST) for i in range(depth_inv_norm.shape[0])])

        images = images.astype(np.float32)
        depths = depths.astype(np.float32)
        poses = poses.astype(np.float32)
        K = K.astype(np.float32)
        
        return {
            'image_paths': image_paths,
            'mask': mask,
            'images': images,
            'depths': depths,
            'poses': poses,
            'K': K,
            'prompt': prompts,
            'depth_inv_norm': depth_inv_norm,
            'depth_inv_norm_small': depth_inv_norm_small,
            'data_load_mode': self.data_load_mode,
        }