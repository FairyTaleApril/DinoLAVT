import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from torchvision import transforms
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
# can set as any tokenizer 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  

def custom_collate_fn(batch): #padding text sequences to the same length
    images = torch.stack([item['image'] for item in batch])  
    segmentations = [torch.tensor(item['segmentation']) for item in batch]  
    bboxes = torch.stack([item['bbox'] for item in batch]) 
    questions = [item['question'] for item in batch]  
    question_masks = [item['question_mask'] for item in batch]  
    answers = [item['answer'] for item in batch]  
    answer_masks = [item['answer_mask'] for item in batch]  
    file_names = [item['file_name'] for item in batch]
    iscrowds = torch.stack([item['iscrowd'] for item in batch])

    questions_padded = pad_sequence(questions, batch_first=True, padding_value=tokenizer.pad_token_id)
    question_masks_padded = pad_sequence(question_masks, batch_first=True, padding_value=0) 
    answers_padded = pad_sequence(answers, batch_first=True, padding_value=tokenizer.pad_token_id)
    answer_masks_padded = pad_sequence(answer_masks, batch_first=True, padding_value=0)
    return {
        'image': images, #tensor
        'question': questions_padded, #tensor
        'question_mask': question_masks_padded,#tensor
        'answer': answers_padded,#tensor
        'answer_mask': answer_masks_padded,#tensor
        'segmentation': segmentations, # 2-dim list
        'bbox': bboxes,#tensor
        "file_name": file_names,#tensor
        'iscrowd':iscrowds#tensor
    }


class RefCOCOPlusDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_transforms=None):
        self.hf_dataset = hf_dataset
        self.text_tokenizer = tokenizer #BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = image_transforms

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]

        image = sample['image']

        if self.transform:
            image = self.transform(image)

        question = sample['question']
        answer = sample['answer'][0]

        question_inputs = self.text_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        answer_inputs = self.text_tokenizer(answer, return_tensors="pt", padding=True, truncation=True) 

        return {
            'image': image,
            'question': question_inputs['input_ids'].squeeze(0), 
            'question_mask': question_inputs['attention_mask'].squeeze(0),
            'answer': answer_inputs['input_ids'].squeeze(0),
            'answer_mask': answer_inputs['attention_mask'].squeeze(0),
            'segmentation': torch.tensor(sample['segmentation']),  
            'bbox': torch.tensor(sample['bbox']), 
            'iscrowd': torch.tensor(sample['iscrowd']),
            'file_name': sample['file_name']
        }
    



#########################################
# test_code
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# dataset = load_dataset("lmms-lab/RefCOCOplus", split='val')


# train_dataset = RefCOCOPlusDataset(dataset, tokenizer, image_transforms=transform)


# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)


# for batch in train_loader:
#     images = batch['image']  
#     questions = batch['question'] 
#     answers = batch['answer']  
#     segmentations = batch['segmentation']  
#     bboxes = batch['bbox']  
#     file_names = batch['file_name']  

#     print(images.shape) 
#     print(questions.shape)  
#     print(answers.shape)  
#     print(segmentations)
#     print(bboxes.shape)  
#     break 
