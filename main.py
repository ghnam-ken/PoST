import argparse

def get_options():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='val')

    # data
    parser.add_argument('--data', type=str, required=True, help='name of data to load')
    parser.add_argument('--data_root', type=str, required=True, help='path/to/data/root')
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--num_cp', type=int, default=128, help='the number of control pointsn to sample')
    parser.add_argument('--img_size', type=int, default=480)

    # model
    parser.add_argument('--num_iter', type=int, default=5)
    parser.add_argument('--net_path', type=str, default=None)

    # others
    parser.add_argument('--save_root', type=str, default='./results')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = get_options()
    if opt.mode == 'val':
        from val import main
    else:
        raise NotImplementedError(f'{opt.mode} is not implemented yet')

    main(opt)

    
    