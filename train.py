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
from transformers import GPT2Tokenizer, BertModel
import numpy as np
from pycocotools import mask
from PIL import Image

from utils.args_parser import get_args_parser
from utils.logger import info, Logger, error
import utils.util as util
from models.lavt import Lavt

from data.dataset import MyDataset
from data.dataset_hug import RefCOCOPlusDataset


def criterion(input, target, device):
    weight = torch.FloatTensor([0.9, 1.1]).to(device)
    return nn.functional.cross_entropy(input, target, weight=weight)


def get_dataset(args, image_set, image_transform, target_transforms):
    ds = MyDataset(args, image_transforms=image_transform, target_transforms=target_transforms, split=image_set)

    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer.pad_token = tokenizer.eos_token
    # dataset = load_dataset("lmms-lab/RefCOCOplus", split='val')
    # ds = RefCOCOPlusDataset(dataset, tokenizer, image_transforms=image_transform)

    return ds


def test(model, data_loader: data.DataLoader, device):
    model.eval()

    loss = 0.0
    i = 0
    for _, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing LAVT..."):
        token, target, sentences, attentions, img = data

        numpy_image = target.numpy()[0]
        numpy_image = ((numpy_image + 1) * 127.5).astype(np.uint8)
        imgg = Image.fromarray(numpy_image)
        imgg.show()

        token = token.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sentences = sentences.to(device, non_blocking=True)
        attentions = attentions.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.to(device)
        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)

        output = model(token, embedding, attentions, img)

        output_mask = output.cpu().argmax(1).data
        output_mask = output_mask[0].numpy()
        output_mask = (output_mask * 255.0).astype(np.uint8)
        output_img1 = Image.fromarray(output_mask)
        output_img1.show()

        img_np = img.cpu()[0].permute(1, 2, 0).numpy()
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.show()

        # output = ((output.cpu().detach().numpy() + 1) * 127.5).astype(np.uint8)
        # img1 = Image.fromarray(output[0][0])
        # img1.show()
        # img2 = Image.fromarray(output[0][1])
        # img2.show()

        # loss = criterion(output, target, device)
        # loss += loss.item()
        # i += 1

        gc.collect()
        torch.cuda.empty_cache()

    # return {'test_loss': loss / i}


def train_one_epoch(model, criterion, optimizer, data_loader: data.DataLoader, device):
    model.train()

    loss = 0.0
    i = 0
    for _, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training LAVT..."):
        token, target, sentences, attentions, img = data

        token = token.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sentences = sentences.to(device, non_blocking=True)
        attentions = attentions.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.to(device)
        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)

        output = model(token, embedding, attentions, img)

        loss = criterion(output, target, device)
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

    util.seed_everything(args.seed)
    util.print_gpu_info()

    Logger(args)

    os.makedirs(args.ckpt_output_dir, exist_ok=True)

    device = args.device

    info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    info("{}".format(args).replace(', ', ',\n'))

    image_transform = transforms.Compose([
        # transforms.Resize(args.img_size, args.img_size),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    train_ds = get_dataset(args, "train", image_transform=image_transform, target_transforms=target_transforms)
    # dataset_test = get_dataset("val", transform=transform, args=args)

    train_dl = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.drop_shuffle,
        drop_last=args.drop_shuffle)
        # pin_memory=args.pin_mem)
    # num_workers=int(args.num_workers))

    model = Lavt(args)
    model.to(device)
    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # tb_writer = None
    # if args.tb_dir is not None:
    #     tb_writer = SummaryWriter(log_dir=args.tb_dir)

    # training
    info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        info(f"Now epoch: {epoch}")

        train_info = train_one_epoch(model, criterion, optimizer, train_dl, device)
        # info(f'Train info:')
        # for k, v in train_info.items():
        #     # if tb_writer is not None:
        #     #     tb_writer.add_scalar(k, v, epoch)
        #     info(f'{k}: {v}')

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

    test(model, train_dl, device)


if __name__ == '__main__':
    main()
