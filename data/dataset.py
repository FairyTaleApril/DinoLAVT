import os

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from transformers import BertTokenizer

from data.refer import REFER
from models.dinov2 import DINOv2


class MyDataset(data.Dataset):
    def __init__(self, args, image_transforms=None, target_transforms=None, split='train'):
        self.dinov2 = DINOv2()
        self.classes = []
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.split = split
        self.refer = REFER(args.data_dir, args.img_dir, args.dataset, args.splitBy, args.max_image_num)

        self.max_tokens = 20
        self.img_size = args.img_size

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # if we are testing on a dataset, test all sentences of an object;
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

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
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

        img_pil = transforms.ToPILImage()(img)
        inputs = self.dinov2.process_image(img_pil)
        dino_token = self.dinov2.get_tokens(inputs)

        target = Image.fromarray(annot.astype(np.uint8), mode="P")
        target = F.resize(target, [self.img_size, self.img_size], interpolation=Image.NEAREST)
        target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)

        choice_sent = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask = self.attention_masks[index][choice_sent]

        return img, dino_token, target, tensor_embeddings, attention_mask
