import datetime
import gc
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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


def criterion(inputs, target, device):
    weight = torch.FloatTensor([0.9, 1.1]).to(device)
    return nn.functional.cross_entropy(inputs, target, weight=weight)


def get_dataset(args, image_set, image_transform, target_transforms):
    ds = MyDataset(args, image_transforms=image_transform, target_transforms=target_transforms, split=image_set)
    return ds


def test(args, model, bert_model, crit, data_loader, device):
    model.eval()

    with open(args.print_dir + '/test_loss.txt', 'w') as f:
        loss, num = 0.0, 0
        cum_I, cum_U = 0.0, 0.0
        mean_IoU = []
        for i, next_data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing LAVT..."):
            imgs, tokens, targets, sentences, attentions = next_data

            numpy_imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
            for j in range(len(numpy_imgs)):
                numpy_img = numpy_imgs[j]
                numpy_img = ((numpy_img - numpy_img.min()) / (numpy_img.max() - numpy_img.min()) * 255).astype(np.uint8)
                ori_img = Image.fromarray(numpy_img)
                ori_img.save(f'output/img/ori_img{i}-{j}.jpg')
                # ori_img.show()

            numpy_targets = targets.numpy()
            for j in range(len(numpy_targets)):
                numpy_target = ((numpy_targets[j] + 1) * 127.5).astype(np.uint8)
                target_img = Image.fromarray(numpy_target)
                target_img.save(f'output/img/target_img{i}-{j}.jpg')
                # target_img.show()

            imgs = imgs.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sentences = sentences.to(device, non_blocking=True).squeeze(1)
            attentions = attentions.to(device, non_blocking=True).squeeze(1)

            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            embeddings = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)

            with torch.no_grad():
                output = model(imgs, tokens, embeddings, attentions)

            loss = crit(output, targets, device)
            f.write(f"Test {i} loss: {loss}\n")
            loss += loss.item()
            num += 1

            numpy_outputs = output.cpu().argmax(1).data.numpy()
            for j in range(len(numpy_outputs)):
                numpy_output = (numpy_outputs[j] * 255.0).astype(np.uint8)
                output_img = Image.fromarray(numpy_output)
                output_img.save(f'output/img/output_img{i}-{j}.jpg')
                # output_img.show()

            I, U = util.computeIoU(numpy_outputs, numpy_targets)
            this_iou = 0.0 if U == 0 else I * 1.0 / U
            mean_IoU.append(this_iou)
            cum_I += I
            cum_U += U

            gc.collect()
            torch.cuda.empty_cache()

        loss = loss / num
        f.write(f"Total test loss: {loss}\n")

        mIoU = np.mean(np.array(mean_IoU))
        info(f'Test result: mean IoU = {mIoU * 100.: .2f}')
        info(f'Test result: overall IoU = {cum_I * 100. / cum_U: .2f}')
        f.write(f'Mean IoU = {mIoU * 100.: .2f}\n')
        f.write(f'Overall IoU = {cum_I * 100. / cum_U: .2f}\n')


def train_one_epoch(epoch, model, bert_model, crit, optimizer, data_loader, device):
    model.train()

    loss, num = 0.0, 0
    for _, next_data in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Training epoch {epoch}"):
        imgs, tokens, targets, sentences, attentions = next_data

        imgs = imgs.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sentences = sentences.to(device, non_blocking=True).squeeze(1)
        attentions = attentions.to(device, non_blocking=True).squeeze(1)

        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embeddings = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)

        output = model(imgs, tokens, embeddings, attentions)

        loss = crit(output, targets, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss += loss.item()
        num += 1

        gc.collect()
        torch.cuda.empty_cache()

    loss = loss / num
    return {'train_loss': loss}


def main():
    args = get_args_parser()
    os.makedirs(args.tb_dir, exist_ok=True)
    os.makedirs(args.ckpt_output_dir, exist_ok=True)
    os.makedirs(args.img_output_dir, exist_ok=True)
    os.makedirs(args.print_dir, exist_ok=True)
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

    test_ds = get_dataset(args, "test", image_transform=image_transform, target_transforms=target_transforms)
    test_dl = data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=args.drop_shuffle,
        drop_last=args.drop_shuffle,
        pin_memory=args.pin_mem,
        num_workers=int(args.num_workers))

    model = Lavt(args)
    model.to(device)

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        model.load_state_dict(state_dict)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tb_writer = None
    if args.tb_dir is not None:
        tb_writer = SummaryWriter(log_dir=args.tb_dir)

    # training
    info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_info = train_one_epoch(epoch, model, bert_model, criterion, optimizer, train_dl, device)
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

    test(args, model, bert_model, criterion, test_dl, device)


def load_test(pth=None):
    args = get_args_parser()
    os.makedirs(args.img_output_dir, exist_ok=True)
    os.makedirs(args.print_dir, exist_ok=True)
    device = args.device

    if pth is not None:
        args.ckpt = pth

    Logger(args)
    util.seed_everything(args.seed)

    image_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    target_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    test_ds = get_dataset(args, "test", image_transform=image_transform, target_transforms=target_transforms)
    test_dl = data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=args.drop_shuffle,
        drop_last=args.drop_shuffle,
        pin_memory=args.pin_mem,
        num_workers=int(args.num_workers))

    model = Lavt(args)
    model.to(device)

    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)

    test(args, model, bert_model, criterion, test_dl, device)


if __name__ == '__main__':
    main()
    # load_test('output/dinov2 40 depths[2,2,2] num_heads[3,3,3] num_heads_fusion[1,1,1]/ckpt/lavt_base_40.pth')
    # load_test('output/ckpt/lavt_base_5.pth')
