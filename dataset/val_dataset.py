from os import stat
from torch.utils.data import Dataset
import numpy as np
import cv2

from PIL import Image

from utils.proc import sample_cnt, sample_cnt_with_idx 


class ValDataset(Dataset):
    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.opt = opt

        if self.dataset.anno_type == 'pointset':
            self.read_anno = self._read_pointset
        elif self.dataset.anno_type == 'mask':
            self.read_anno = self._read_mask
    
    def __getitem__(self, idx):
        seq = self.dataset[idx]

        img_files, anno_files, others = seq
        if 'init_path' in others:
            init_path = others['init_path']

        if 'idx_path' in others:
            idx_path = others['idx_path']
            idx = np.loadtxt(idx_path)
        else:
            idx = None
        
        init = self.read_anno(init_path)

        if not idx is None:
            init, idx = sample_cnt_with_idx(init, idx, self.opt.num_cp)
        else:
            init = sample_cnt(init, self.opt.num_cp)

        imgs = []
        annos = []

        for img_path in img_files:
            img = Image.open(img_path)
            img = np.array(img)
            imgs.append(img)

        for anno_path in anno_files:
            if anno_path:
                anno = self.read_anno(anno_path)
            else:
                anno = []
            annos.append(anno)

        others['img_files'] = img_files
        others['anno_files'] = anno_files 

        return imgs, annos, init, idx, others


    @staticmethod
    def _read_pointset(txt_path):
        return np.loadtxt(txt_path)

    @staticmethod
    def _read_mask(self, png_path):
        mask = Image.open(png_path)
        cnt, _ = cv2.findContours(mask, 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_NONE)
        if not cnt:
            return []

        if len(cnt) > 1 or len(cnt[0]) > 1:
            raise NotImplementedError('more than 1 contour in a mask is not supported')
            
        return cnt[0][0]

    def __len__(self):
        return len(self.dataset)

        



