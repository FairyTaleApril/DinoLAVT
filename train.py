import datetime
import os
import time
import gc
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
from pycocotools import mask

from utils.args_parser import get_args_parser
from utils.logger import info, Logger, error
import utils.util as util
from models.lavt import Lavt

from data.dataset_hug import RefCOCOPlusDataset


def criterion(input, target, device):
    weight = torch.FloatTensor([0.9, 1.1]).to(device)
    return nn.functional.cross_entropy(input, target, weight=weight)


def get_dataset(transform):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("lmms-lab/RefCOCOplus", split='val')
    ds = RefCOCOPlusDataset(dataset, tokenizer, image_transforms=transform)

    return ds

def train_one_epoch(model, criterion, optimizer, data_loader: data.DataLoader,
                    device):
    model.train()

    loss = 0.0
    i = 0
    for _, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training LAVT..."):
        image = data['image']
        sentences = data['answer']
        attentions = data['answer_mask']
        seg = data['segmentation']
        if type(seg[0]) == list:  # polygon
            error('segment is a list in list')
        m = mask.decode(seg)
        m = np.sum(m, axis=2)
        target = m.astype(np.uint8)

        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sentences = sentences.to(device, non_blocking=True)
        attentions = attentions.to(device, non_blocking=True)
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        output = model(image, sentences, l_mask=attentions)

        loss = criterion(output, target, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss += loss.item()
        i += 1

        gc.collect()
        torch.cuda.empty_cache()

    return {'train_loss': loss / i}


def main(args):
    util.seed_everything(args.seed)

    Logger(args)

    util.print_gpu_info()
    device = args.device

    info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    info("{}".format(args).replace(', ', ',\n'))

    transform = transforms.Compose([
        transforms.Resize(args.img_size, args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_ds = get_dataset("train", transform=None, args=args)  # TODO: define transform
    # dataset_test = get_dataset("val", transform=transform, args=args)

    train_dl = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.drop_shuffle,
        drop_last=args.drop_shuffle)
        # pin_memory=args.pin_mem,
        # num_workers=int(args.num_workers))

    model = Lavt(args)
    model.to(device)
    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tb_writer = None
    # if args.tb_dir is not None:
    #     tb_writer = SummaryWriter(log_dir=args.tb_dir)

    # training
    info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        info(f"Now epoch: {epoch}")

        train_info = train_one_epoch(model, criterion, optimizer, train_dl, device)
        info(f'Train info:')
        for k, v in train_info.items():
            # if tb_writer is not None:
            #     tb_writer.add_scalar(k, v, epoch)
            info(f'{k}: {v}')

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_filename = f'{args.model}_{epoch}.pth'
            path = os.path.join(args.ckpt_output_dir, save_filename)
            torch.save(model.cpu().state_dict(), path)
            info(f'Save the model to {path}')
            model.to(device)

    # if tb_writer is not None:
    #     tb_writer.flush()
    #     tb_writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    info(f'Training time {total_time_str}')


if __name__ == '__main__':
    args = get_args_parser()

    main(args)
