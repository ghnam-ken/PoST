# Polygonal Point Set Tracking
Accepted to [CVPR2021](http://cvpr2021.thecvf.com/)
\[[paper](https://arxiv.org/abs/2105.14584)\]
\[[PoST](https://drive.google.com/file/d/1yNhiCtnWpzYZuuRGR886WZ48Fpwt-EhY/view?usp=sharing)\]
\[[results](https://drive.google.com/file/d/1gElr0mvIivMHrzj3etMAssNZzGZOmfin/view?usp=sharing)\]

## Abstract
In this paper, we propose a novel learning-based polygonal point set tracking method.
Compared to existing video object segmentation(VOS) methods that propagate pixel-wise object mask information, we propagate a polygonal point set over frames. 
Specifically, the set is defined as a subset of points in the target contour, and our goal is to track corresponding points on the target contour.
Those outputs enable us to apply various visual effects such as motion tracking, part deformation, and texture mapping.
To this end, we propose a new method to track the corresponding points between frames by the global-local alignment with delicately designed losses and regularization terms.
We also introduce a novel learning strategy using synthetic and VOS datasets that makes it possible to tackle the problem without developing the point correspondence dataset.
Since the existing datasets are not suitable to validate our method, we build a new polygonal point set tracking dataset and demonstrate the superior performance of our method over the baselines and existing contour-based VOS methods.
In addition, we present visual-effects applications of our method on part distortion and text mapping.

## Evaluation
1. Download [PoST dataset](https://drive.google.com/file/d/1yNhiCtnWpzYZuuRGR886WZ48Fpwt-EhY/view?usp=sharing)
2. Download [Results](https://drive.google.com/file/d/1gElr0mvIivMHrzj3etMAssNZzGZOmfin/view?usp=sharing) or prepare your own result.
```
cd eval
python eval.py --result_path RESULT_PATH --data_path PoST_PATH --threshs 0.16 0.08 0.04
```

## Notice
**Model Code** will be uploaded soon

## Citation
TBA

