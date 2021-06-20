from PIL import Image, ImageStat
import cv2
import numpy as np
import math
import torch
import torch.nn.functional as F



def rel2abs(cnt, size):
    cnt = cnt/2 + 0.5
    cnt[..., 0] *= size[0]
    cnt[..., 1] *= size[1]
    return cnt

def abs2rel(cnt, size):
    cnt[..., 0] /= size[0]
    cnt[..., 1] /= size[1]
    cnt = (cnt-0.5)*2 # [-1 1]
    return cnt

def cnt2mask(cnt, size):
    B, N, _, _ = cnt.shape

    cnt = rel2abs(cnt.clone(), size)
    masks = []
    for i in range(B):
        mask = np.zeros((size[1], size[0], 3))
        c = cnt[i].data.cpu().numpy().astype('int32')
        mask = cv2.drawContours(mask, [c], -1, (1, 1, 1), cv2.FILLED)
        mask = torch.FloatTensor(np.mean(mask, axis=2)) # H, W
        masks.append(mask.unsqueeze(0)) # 1, H, W
    masks = torch.stack(masks, dim=0) # B, 1, H, W
    masks = masks.cuda() if torch.cuda.is_available() else masks
    return masks

def transform(img, cnt, theta):
    img_align = transform_img(img, theta)
    cnt_align = transform_cnt(cnt, theta)
    return img_align, cnt_align

def transform_img(img, theta):
    inv_theta = invert(theta)
    grid = F.affine_grid(inv_theta, img.shape)
    img_align = F.grid_sample(img, grid)
    return img_align

def transform_cnt(cnt, theta):
    # cnt: B, N, 1, 2
    # theta: B, 2, 3

    B, N, _, _ = cnt.shape

    cnt = cnt.clone()
    cnt_align = torch.cat((cnt, torch.ones_like(cnt)[...,:1]), dim=3) # B, N, 1, 3 [y,x,1]
    theta = theta.transpose(1, 2) # B, 3, 2
    theta = theta.unsqueeze(1) # B, 1, 3, 2
    cnt_align = torch.matmul(cnt_align, theta) # B, N, 1, 2

    return cnt_align

def cnt2poly(cnt):
    x_min = torch.min(cnt[...,0], dim=1, keepdim=True)[0]
    y_min = torch.min(cnt[...,1], dim=1, keepdim=True)[0]
    poly = cnt.clone()
    poly[...,0] -= x_min
    poly[...,1] -= y_min
    return poly

def get_adj_ind(n_adj, n_nodes):
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0])
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    ind = ind.cuda() if torch.cuda.is_available() else ind
    return ind

def erode(mask, it=10):
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=it)
    return mask

def dilate(mask, it=10):
    if it > 0:
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=it)
    return mask

def sample_cnt(cnt, num_cp):
    # cnt: N_all, 2
    cnt = cnt.clone()
    N_all = cnt.shape[0]

    select = np.linspace(0, N_all-(N_all//num_cp), num_cp, dtype=np.int64)
    sample = cnt[select] # N, 2
    return sample

def sample_cnt_with_idx(cnt, idx, num_cp):
    total_num = cnt.shape[0]
    select = np.linspace(0, total_num-1, num_cp).astype(np.int64)
    sampled_idx = ((idx/total_num)*num_cp).astype(np.int64)
    select[sampled_idx] = idx
    sampled_cnt = cnt[select] # N, 2
    sampled_cnt = sampled_cnt.reshape(-1, 1, 2)
    
    return sampled_cnt, sampled_idx

def update_cnt(cnt_out, cnt_smpl, start):
    B = cnt_out.shape[0]
    N = cnt_out.shape[1]
    N_smpl = cnt_smpl.shape[1]

    skip = N // N_smpl

    cnt_out[:,start::skip]= cnt_smpl

    return cnt_out

def upsample_offset(offset, stride=2):
    B, N, _, _ = offset.shape

    offset_up = offset.clone()
    offset_up = offset_up.expand(B, N, stride, 2).contiguous().view(B, N*stride, 1, 2) # B, N*stride, 1, 2
    offset_up = torch.cat([offset_up.roll(i, dims=1) for i in range(stride)], dim=2) # B, N*stride, stride, 2 
    offset_up = torch.mean(offset_up, dim=2, keepdim=True) # B, N*stride, 1, 2)
    return offset_up

def invert(theta):
    inv_theta = torch.zeros_like(theta)
    det = theta[:,0,0]*theta[:,1,1] - theta[:,0,1]*theta[:,1,0]
    adj_x = -theta[:,1,1]*theta[:,0,2] + theta[:,0,1]*theta[:,1,2]
    adj_y = theta[:,1,0]*theta[:,0,2] - theta[:,0,0]*theta[:,1,2]
    inv_theta[:,0,0] = theta[:,1,1]
    inv_theta[:,1,1] = theta[:,0,0]
    inv_theta[:,0,1] = -theta[:,0,1]
    inv_theta[:,1,0] = -theta[:,1,0]
    inv_theta[:,0,2] = adj_x
    inv_theta[:,1,2] = adj_y
    inv_theta = inv_theta / det.view(-1, 1, 1)

    return inv_theta

def generate_pos_emb(num_pos):
    emb = np.arange(0, num_pos, 1, dtype=np.float) * 2 * math.pi / num_pos
    sin_p = np.sin(emb) # N
    cos_p = np.cos(emb) # N
    emb = np.stack([sin_p, cos_p], axis=0) # 2, N
    emb = torch.FloatTensor(emb)
    return emb 

def crop_and_resize(img, cbox, context, size, 
                    margin=0, correct=False, resample=Image.BILINEAR):

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    assert(isinstance(img, Image.Image))
    assert(isinstance(cbox, np.ndarray))
    min_sz = 10

    W, H = img.size

    if img.mode != 'P':
        img = img.convert('RGB')

    cbox = cbox.copy()

    cbox[0] = cbox[0] + 0.5 * (cbox[2]-1)
    cbox[1] = cbox[1] + 0.5 * (cbox[3]-1)

    # define output format
    out_size = size + margin

    if img.mode != 'P':
        avg_color = ImageStat.Stat(img).mean
        avg_color = tuple(int(round(c)) for c in avg_color)
        patch = Image.new(img.mode, (out_size, out_size), color=avg_color)
    else:
        patch = Image.new(img.mode, (out_size, out_size))

    m = (cbox[2] + cbox[3])*0.5
    search_range = math.sqrt((cbox[2]+m) * (cbox[3]+m)) * context

    # crop
    crop_sz = int(round(search_range * out_size / size))
    crop_sz = max(5, crop_sz)
    crop_ctr = cbox[:2]
    
    dldt = crop_ctr - 0.5*(crop_sz-1)
    drdb = dldt + crop_sz
    plpt = np.maximum(0, -dldt)
    dldt = np.maximum(0, dldt)
    drdb = np.minimum((W, H), drdb)

    dltrb = np.concatenate((dldt, drdb))
    dltrb = np.round(dltrb).astype('int')
    cp_img = img.crop(dltrb)

    # resize
    cW, cH = cp_img.size
    tW = max(cW * out_size / crop_sz, min_sz)
    tH = max(cH * out_size / crop_sz, min_sz)
    tW = int(round(tW))
    tH = int(round(tH))
    rz_img = cp_img.resize((tW, tH), resample)

    # calculate padding to paste to patch
    plpt = plpt * out_size / crop_sz
    plpt_ = np.round(plpt).astype('int')
    pltrb = np.concatenate((plpt_, plpt_ + (tW, tH)))

    # paste
    patch.paste(rz_img, pltrb)

    # if flag 'correct' is Ture, return information about turning back.
    if correct:
        scale = crop_sz / out_size
        leftmost = crop_ctr - (crop_sz - 1) / 2
        return patch, leftmost, scale

    return patch