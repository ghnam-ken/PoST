from dataset import PoST, val_dataset
from torch.utils.data import DataLoader
from models.model import PoSTModel

import tqdm
import os 
import numpy as np


data_factory = {
    "PoST": PoST.PoST
}

def get_save_name(fname):
    save_name = os.path.basename(fname)
    save_name = ''.join(save_name.split('.')[:-1]) + '.txt'
    return save_name

def main(opt):
    if opt.data in data_factory:
        data = data_factory[opt.data]
        data = data(opt.data_root)
    
    val_data = val_dataset.ValDataset(data, opt)
    val_dataloader = DataLoader(val_data,
                                batch_size=1,
                                num_workers=opt.num_worker,
                                shuffle=False,
                                collate_fn=lambda x: x[0])
    model = PoSTModel(opt)

    if os.path.exists(opt.save_root):
        print(f"[WARNING] save root '{opt.save_root}' already exists. results will be replaced")
        

    for i, (imgs, annos, init, idx, others) \
                            in enumerate(val_dataloader):
        if 'seq' in others:
            seq = others['seq']
        else: 
            raise ValueError('sequence name is not identified')
        
        img_files = others['img_files']

        outputs = {}
        save_dir = os.path.join(opt.save_root, seq) 
        
        print(f'[VAL-{seq}] start inference')
        for j, img in enumerate(tqdm.tqdm(imgs)):
            if j == 0:
                print(f'[VAL-{seq}] initialize')
                cnt = model.initialize(imgs[j], init)
            else:
                cnt = model.inference(img)

            if len(idx) > 0:
                cnt = cnt[idx]

            save_name = get_save_name(img_files[j])
            outputs[save_name] = cnt

        save_dir = os.path.join(opt.save_root, seq) 
        os.makedirs(save_dir, exist_ok=True)

        print(f'[VAL-{seq}] save outputs')
        for save_name, cnt in outputs.items():
            save_path = os.path.join(save_dir, save_name)
            np.savetxt(save_path, cnt)
