import torch
import torch.nn as nn

import numpy as np

from .networks  import PointSetTracker
from utils.proc import sample_cnt, crop_and_resize 


class PoSTModel(object):
    def __init__(self, opt):
        self.opt = opt
        self.net = PointSetTracker(num_iter=opt.num_iter)
        print('[MODEL] initialize point set tracker')
        if opt.net_path:
            self.load(opt.net_path)

        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net)
           
    def initialize(self, img, cnt):
        self.frame_num = 0
        cnt = torch.FloatTensor(cnt).view(1, -1, 1, 2)
        cnt = self._sample_cnt_batch(cnt, self.opt.num_cp)
        curr_patch, _, _ = self._preprocess_img(img, cnt)

        # cnt = cnt.cuda()
        self.rnn_fs, self.pos_emb = self.net(cnt)

        self.prev_patch = curr_patch
        self.prev_cnt = cnt

        return self._make_output(cnt)
    
    def inference(self, img):
        self.frame_num += 1
        curr_patch, prev_cnt, leftmost, scale = self._preprocess(img, self.prev_cnt)
        
        with torch.set_grad_enabled(False):
            self.net.eval()
            curr_cnt, rnn_fs = self.net(self.prev_patch, curr_patch,
                                        prev_cnt, self.pos_emb, self.rnn_fs)
        
        curr_cnt = self._postprocess_cnt(curr_cnt, leftmost, scale)

        self.prev_patch, _, _ = self._preprocess_img(img, curr_cnt)
        self.rnn_fs = rnn_fs
        self.prev_cnt = curr_cnt

        return self._make_output(curr_cnt)

    def load(self, net_path):
        state_dict = torch.load(net_path)
        self.net.load_state_dict(state_dict)
        print(f'[MODEL] load model weight from {net_path}')

    @staticmethod
    def _postprocess_cnt(cnt, leftmost, scale):
        cnt *= scale
        cnt[...,0] += leftmost[0]
        cnt[...,1] += leftmost[1]
        return cnt

    def _preprocess(self, img, cnt):
        img, leftmost, scale = self._preprocess_img(img, cnt)
        cnt = self._preprocess_cnt(cnt, leftmost, scale)
        return img, cnt, leftmost, scale 

    @staticmethod
    def _preprocess_cnt(cnt, leftmost, scale):
        cnt[...,0] -= leftmost[0]
        cnt[...,1] -= leftmost[1]
        cnt /= scale
        return cnt

    def _preprocess_img(self, img, cnt):
        dsize = self.opt.img_size

        box = self._cnt2box(cnt)
        img, leftmost, scale = crop_and_resize(img, box, context=1, size=dsize, correct=True)
        img = np.array(img)
        # to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.

        return img, leftmost, scale

    def _sample_cnt_batch(self, cnt, num_cp):
        # cnt: B, N_all, 1, 2
        B = cnt.shape[0]
        samples = []        

        for b in range(B):
            c = cnt[b]
            sample = sample_cnt(c, num_cp)
            samples.append(sample)
        samples = torch.stack(samples, dim=0)
        return samples

    @staticmethod
    def _cnt2box(cnt):
        cnt_ = cnt.data.cpu().numpy()
        xmin = np.min(cnt_[...,0])
        xmax = np.max(cnt_[...,0])
        ymin = np.min(cnt_[...,1])
        ymax = np.max(cnt_[...,1])

        w = xmax - xmin + 1
        h = ymax - ymin + 1

        return np.asarray([xmin, ymin, w, h])

    @staticmethod
    def _make_output(cnt):
        cnt = cnt.view(-1, 2)
        cnt = cnt.data.cpu().numpy().astype(np.int)
        return cnt
