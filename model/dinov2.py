from transformers import AutoImageProcessor, Dinov2Model
import torch
from datasets import load_dataset


class DINOv2:
    def __init__(self, model_name="facebook/dinov2-base"):
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)

    def process_image(self, images):
        inputs = self.image_processor(images, return_tensors="pt")
        return inputs

    def get_cls_tokens(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        return cls_tokens


def main():
    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]

    dinov2 = DINOv2()
    inputs = dinov2.process_image(image)
    cls_token = dinov2.get_cls_tokens(inputs)
    print(cls_token.shape)


if __name__ == "__main__":
    main()
