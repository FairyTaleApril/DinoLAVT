import os
import time

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from transformers import BertTokenizer
from tqdm import tqdm

from data.refer import REFER
from models.dinov2 import DINOv2


class MyDataset(data.Dataset):
    def __init__(self, args, image_transforms=None, target_transforms=None, split='train'):
        print('Constructing dataset...')
        tic = time.time()

        self.max_tokens = 20
        self.img_size = args.img_size
        self.img_embed_model = args.img_embed_model
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.split = split

        self.dinov2 = DINOv2() if self.img_embed_model == 'dinov2' else None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.refer = REFER(args.data_dir, args.img_dir, args.dataset, args.splitBy, args.max_image_num)
        ref_ids = self.refer.getRefIds(split=self.split)
        self.ref_ids = ref_ids

        self.classes = []
        self.input_ids = []
        self.pre_attention_masks = []

        # If we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # Truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.pre_attention_masks.append(attentions_for_ref)

        self.imgs = []
        self.dino_tokens = []
        self.targets = []
        self.tensor_embeddings = []
        self.attention_masks = []

        self.process_data()

        print('Dataset constructed')

    def process_data(self):
        length = len(self)
        for i in tqdm(range(length), desc="Processing data"):
            this_ref_id = self.ref_ids[i]
            this_img_id = self.refer.getImgIds(this_ref_id)
            ref = self.refer.loadRefs(this_ref_id)
            ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
            annot = np.zeros(ref_mask.shape)
            annot[ref_mask == 1] = 1

            this_img = self.refer.Imgs[this_img_id[0]]
            img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
            img = F.resize(img, [self.img_size, self.img_size])
            img = F.to_tensor(img)
            img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.imgs.append(img)

            if self.img_embed_model == 'dinov2':
                img_pil = transforms.ToPILImage()(img)
                inputs = self.dinov2.process_image(img_pil)
                dino_token = self.dinov2.get_tokens(inputs)
            else:
                dino_token = 0
            self.dino_tokens.append(dino_token)

            target = Image.fromarray(annot.astype(np.uint8), mode="P")
            target = F.resize(target, [self.img_size, self.img_size], interpolation=Image.NEAREST)
            target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)
            self.targets.append(target)

            choice_sent = np.random.choice(len(self.input_ids[i]))
            tensor_embedding = self.input_ids[i][choice_sent]
            self.tensor_embeddings.append(tensor_embedding)
            attention_mask = self.pre_attention_masks[i][choice_sent]
            self.attention_masks.append(attention_mask)

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        img = self.imgs[index]
        dino_token = self.dino_tokens[index]
        target = self.targets[index]
        tensor_embedding = self.tensor_embeddings[index]
        attention_mask = self.attention_masks[index]
        return img, dino_token, target, tensor_embedding, attention_mask
