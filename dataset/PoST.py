import os
import glob
import six


class PoST(object):
    def __init__(self, root_dir):
        super(PoST, self).__init__()
        self.name = 'PT'
        self.anno_type = 'pointset'
        self.root_dir = root_dir

        self.img_dirname = 'images'
        self.anno_dirname = 'points'
        self.idx_filename = 'indices.txt'
        self.init_filename = 'init.txt'

        self.seq_list = sorted(os.listdir(self.root_dir))

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_list:
                raise Exception('Sequence {} not found.'.format(index))
            idx = self.seq_list.index(index)
        else:
            idx = index%len(self.seq_list)

        seq_name = self.seq_list[idx]

        img_path = os.path.join(self.root_dir, seq_name, self.img_dirname)
        img_files = sorted(glob.glob(os.path.join(img_path, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))

        anno_files = []
        for img_path in img_files:
            anno_file_name = os.path.basename(img_path).replace('.jpg', '.txt')
            anno_path = os.path.join(self.root_dir, self.seq_list[idx], self.anno_dirname, anno_file_name)
            if os.path.isfile(anno_path):
                anno_files.append(anno_path)
            else:
                anno_files.append(None)

        init_path = os.path.join(self.root_dir, self.seq_list[idx], self.init_filename)
        idx_path = os.path.join(self.root_dir, self.seq_list[idx], self.idx_filename)

        others = {}
        others['init_path'] = init_path
        others['idx_path'] = idx_path
        others['seq'] = seq_name
        return img_files, anno_files, others

    def __len__(self):
        return len(self.seq_list)
