# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json


class CPDataset(data.Dataset):
    """Dataset for CP-VTON+.
    """

    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.stage = opt.stage  # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

       def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        
        if self.stage == 'GMM':
            c = Image.open(osp.join(self.data_path, 'cloth', c_name)).convert("RGB")
            cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name)).convert('L')
        else:
            c = Image.open(osp.join(self.data_path, 'warp-cloth', im_name)).convert("RGB")
            cm = Image.open(osp.join(self.data_path, 'warp-mask', im_name)).convert('L')
        
        c = self.transform(c)
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array).unsqueeze(0)
        
        im = Image.open(osp.join(self.data_path, 'image', im_name)).convert("RGB")
        im = self.transform(im)
        
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse-new', parse_name)).convert('L')
        parse_array = np.array(im_parse)
        
        im_mask = Image.open(osp.join(self.data_path, 'image-mask', parse_name)).convert('L')
        mask_array = np.array(im_mask)
        parse_shape = (mask_array > 0).astype(np.float32)
        
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32)
        
        parse_shape_ori = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape_ori = parse_shape_ori.convert("RGB")  # Ensure it is in RGB
        
        parse_shape = parse_shape_ori.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        parse_shape_ori = parse_shape_ori.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        
        shape_ori = self.transform(parse_shape_ori)
        shape = self.transform(parse_shape)
        pcm = torch.from_numpy(parse_cloth).unsqueeze(0)
        
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = np.array(pose_label['people'][0]['pose_keypoints']).reshape((-1, 3))
        
        pose_map = torch.zeros(pose_data.shape[0], self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        
        for i in range(pose_data.shape[0]):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx, pointy = pose_data[i, 0], pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            pose_map[i] = self.transform(one_map)[0]
        
        im_pose = self.transform(im_pose)
        agnostic = torch.cat([shape, im_pose, pose_map], 0)
        
        im_g = self.transform(Image.open('grid.png')) if self.stage == 'GMM' else ''
        
        result = {
            'c_name': c_name,
            'im_name': im_name,
            'cloth': c,
            'cloth_mask': cm,
            'image': im,
            'agnostic': agnostic,
            'shape': shape,
            'pose_image': im_pose,
            'grid_image': im_g,
            'shape_ori': shape_ori,
            'parse_cloth_mask': pcm,
        }
        
        return result
    def __len__(self):
        return len(self.im_names)


class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(
                train_sampler is None),
            num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)

    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d'
          % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed
    embed()
