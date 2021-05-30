import cv2 
import numpy as np
import math

COLORS = [(252, 192, 116), 
          (30, 201, 130), # LIME 6
          (194, 113, 25), # TEAL7
          (5, 176, 250), # YELLOW 6
          (82, 82, 250), # PINK 5
          (232, 72, 112), # VIOLET 7
          (120, 166, 12)] # TEAL7

def normalize(pts, width, height):
    pts = pts.copy()
    pts[...,0] /= width
    pts[...,1] /= height
    return pts

def denormalize(pts, width, height):
    pts = pts.copy()
    pts[...,0] *= width
    pts[...,1] *= height
    return pts

def draw_pnt(img, pts, colors=None):
    height, width = img.shape[:2]
    avg = math.sqrt(height*width)
    radius = int(0.03*avg)

    if colors is None:
        colors = COLORS

    for i, pt in enumerate(pts):
        stroke_radius = max(1, 5*radius//3)
        main_radius = max(1, 4*radius//4)
        pt = tuple(pt.astype(np.int))

        cv2.circle(img, pt, radius=stroke_radius, color=(255, 255, 255), thickness=-1)
        cv2.circle(img, pt, radius=main_radius, color=colors[i%len(colors)], thickness=-1)

    return img

def sample_cnt(cnt, idx, num):
    total_num = cnt.shape[0]
    select = np.linspace(0, total_num-1, num).astype('int64')
    sampled_idx = ((idx/total_num)*num).astype('int64')

    return sampled_idx

def build_mask_from_points(points, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    points = points.astype(np.int)
    points = points.reshape(1, -1, 1, 2)
    points = tuple(points)
    mask = cv2.drawContours(mask, points, 0, color=(255, 255, 255), thickness=-1)

    return mask

def blend_mask(img, mask, alpha=0.5, color=(0, 255, 255)):
    assert all([isinstance(ins, np.ndarray) for ins in [img, mask]])
    width, height = img.shape[:2]

    mask = cv2.resize(mask, (height, width))
    mask_ = np.zeros_like(img)
    mask_[mask > 0] = color
    blended = img * alpha + mask_ * (1-alpha)
    blended = blended.astype(np.int)
    return blended