import datetime
import gc
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from transformers import BertModel

import utils.util as util
from data.dataset import MyDataset
from models.lavt import Lavt
from utils.args_parser import get_args_parser
from utils.logger import info, Logger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def criterion(inputs, target, device):
    weight = torch.FloatTensor([0.9, 1.1]).to(device)
    return nn.functional.cross_entropy(inputs, target, weight=weight)


def get_dataset(args, image_set, image_transform, target_transforms):
    ds = MyDataset(args, image_transforms=image_transform, target_transforms=target_transforms, split=image_set)
    return ds


def test(model, bert_model, data_loader, device):
    model.eval()

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing LAVT..."):
        img, token, target, sentences, attentions = data

        numpy_img = img.cpu()[0].permute(1, 2, 0).numpy()
        numpy_img = ((numpy_img - numpy_img.min()) / (numpy_img.max() - numpy_img.min()) * 255).astype(np.uint8)
        ori_img = Image.fromarray(numpy_img)
        ori_img.save(f'output/img/output_image{i}.jpg')
        # ori_img.show()

        numpy_targets = target.numpy()
        for j in range(len(numpy_targets)):
            numpy_target = ((numpy_targets[j] + 1) * 127.5).astype(np.uint8)
            target_img = Image.fromarray(numpy_target)
            target_img.save(f'output/img/target_img{i}-{j}.jpg')
            # target_img.show()

        token = token.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sentences = sentences.to(device, non_blocking=True)
        attentions = attentions.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)

        output = model(token, embedding, attentions, img)

        numpy_outputs = output.cpu().argmax(1).data.numpy()
        for j in range(len(numpy_outputs)):
            numpy_output = (numpy_outputs[j] * 255.0).astype(np.uint8)
            output_img = Image.fromarray(numpy_output)
            output_img.save(f'output/img/output_img{i}-{j}.jpg')
            # output_img.show()

        gc.collect()
        torch.cuda.empty_cache()


def train_one_epoch(model, bert_model, crit, optimizer, data_loader: data.DataLoader, device):
    model.train()

    loss = 0.0
    i = 0
    for _, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training LAVT..."):
        img, token, target, sentences, attentions = data

        token = token.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sentences = sentences.to(device, non_blocking=True)
        attentions = attentions.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)

        output = model(token, embedding, attentions, img)

        loss = crit(output, target, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss += loss.item()
        i += 1

        gc.collect()
        torch.cuda.empty_cache()

    return {'train_loss': loss / i}


def main():
    args = get_args_parser()
    os.makedirs(args.ckpt_output_dir, exist_ok=True)
    os.makedirs(args.img_output_dir, exist_ok=True)
    device = args.device

    Logger(args)
    util.seed_everything(args.seed)
    # util.print_gpu_info()

    # info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # info("{}".format(args).replace(', ', ',\n'))

    image_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    target_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    train_ds = get_dataset(args, "train", image_transform=image_transform, target_transforms=target_transforms)

    train_dl = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.drop_shuffle,
        drop_last=args.drop_shuffle,
        pin_memory=args.pin_mem,
        num_workers=int(args.num_workers))

    model = Lavt(args)
    model.to(device)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tb_writer = None
    if args.tb_dir is not None:
        tb_writer = SummaryWriter(log_dir=args.tb_dir)

    # training
    info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        info(f"Now epoch: {epoch}")

        train_info = train_one_epoch(model, bert_model, criterion, optimizer, train_dl, device)
        info(f'Train info:')
        for k, v in train_info.items():
            if tb_writer is not None:
                tb_writer.add_scalar(k, v, epoch)
            info(f'{k}: {v}')

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_filename = f'{args.model}_{epoch}.pth'
            path = os.path.join(args.ckpt_output_dir, save_filename)
            torch.save(model.cpu().state_dict(), path)
            info(f'Save the model to {path}')
            model.to(device)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    info(f'Training time {total_time_str}')

    test(model, bert_model, train_dl, device)


if __name__ == '__main__':
    main()
