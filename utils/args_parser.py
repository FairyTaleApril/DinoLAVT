import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('LAVT with Dino', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU')
    parser.add_argument('--start_epoch', default=1, type=int, help='start training epoch')
    parser.add_argument('--epochs', default=40, type=int, help='total training epochs')
    parser.add_argument('--save_freq', default=20, type=int, help='saving model frequency(epoch)')
    # parser.add_argument('--eval_freq', default=40, type=int, help='evaluation frequency(epoch)')
    parser.add_argument('--tb_dir', default='output/tb', type=str, help='directory of tensorboard log')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Dataset parameters
    parser.add_argument('--max_image_num', default=1, type=int, help='number of images for training')
    parser.add_argument('--data_dir', default='data', type=str, help='directory of dataset')
    parser.add_argument('--img_dir', default=r'E:\ANU\train2014\train2014', type=str, help='directory of images')
    parser.add_argument('--dataset', default='refcoco+', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
    # parser.set_defaults(pin_mem=True)
    # parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--drop_shuffle', action='store_true', default=True,
                        help='drop last and shuffle in training DataLoader')
    # parser.set_defaults(drop_shuffle=True)
    # parser.add_argument('--no_drop_shuffle', action='store_false', dest='drop_shuffle')

    # Model parameters
    parser.add_argument('--model', default='lavt_base', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--ckpt', type=str, default=None, help='where to load model checkpoint')
    parser.add_argument('--ckpt_output_dir', type=str, default='output/ckpt', help='where to save checkpoint')
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR', help='learning rate (absolute lr)')

    return parser.parse_args()
