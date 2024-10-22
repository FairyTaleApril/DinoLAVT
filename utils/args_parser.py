import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('LAVT with Dino', add_help=False)

    parser.add_argument('--batch_size', default=10, type=int, help='Batch size per GPU')
    parser.add_argument('--start_epoch', default=1, type=int, help='Start training epoch')
    parser.add_argument('--epochs', default=40, type=int, help='Total training epochs')
    parser.add_argument('--save_freq', default=20, type=int, help='Saving model frequency(epoch)')
    # parser.add_argument('--eval_freq', default=40, type=int, help='Evaluation frequency(epoch)')
    parser.add_argument('--tb_dir', default='output/tb', type=str, help='Directory of tensorboard log')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use for training / testing')

    # Dataset parameters
    parser.add_argument('--max_image_num', default=200, type=int, help='Number of images for training')
    parser.add_argument('--data_dir', default='data', type=str, help='Directory of dataset')
    parser.add_argument('--img_dir', default=r'E:\ANU\train2014\train2014', type=str, help='Directory of images')
    parser.add_argument('--img_size', default=480, type=int, help='Input image size')
    parser.add_argument('--dataset', default='refcoco+', help='Refcoco, refcoco+, or refcocog')
    parser.add_argument('--splitBy', default='unc', help='Change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--drop_shuffle', action='store_true', default=True,
                        help='Drop last and shuffle in training DataLoader')
    parser.set_defaults(drop_shuffle=True)
    parser.add_argument('--no_drop_shuffle', action='store_false', dest='drop_shuffle')

    # Model parameters
    parser.add_argument('--model', default='lavt_base', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument("--img_embed_model", choices=['dinov2', 'patch_embed'], default='dinov2',
                        help='Choose image embedding model')
    parser.add_argument("--img_token_size", default=768, type=int, help='Length of image token')
    parser.add_argument('--ckpt', type=str, default=None, help='Where to load model checkpoint')
    parser.add_argument('--ckpt_output_dir', type=str, default='output/ckpt', help='Where to save checkpoint')
    parser.add_argument('--img_output_dir', type=str, default='output/img', help='Where to save imgs')
    parser.add_argument('--print_dir', type=str, default='output/print', help='Where to save printed info')
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR', help='Learning rate (absolute lr)')

    args = parser.parse_args()

    args.img_token_size = 768 if args.img_embed_model == 'dinov2' else 96

    return args
