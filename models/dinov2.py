from transformers import AutoImageProcessor, Dinov2Model
import torch
from datasets import load_dataset


class DINOv2:
    def __init__(self, model_name="facebook/dinov2-base"):
        """
        Initialize the DINOv2 model and image processor using the specified model name
        """
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)

    def process_image(self, images):
        """
        Process the input images and return tensors suitable for the model.

        Args:
            images: A list of images (can be PIL images or NumPy arrays).

        Returns:
            inputs (batch_size, channels, height, width): Processed image tensors
        """
        inputs = self.image_processor(images, return_tensors="pt")
        return inputs

    def get_tokens(self, inputs):
        """
        Obtain the class tokens from the model's output for the given inputs.

        Args:
            inputs: Processed image tensors.

        Returns:
            outputs (batch_size, hidden_size): A list of tokens.
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
        outputs = outputs.last_hidden_state[:, 1:, :]
        outputs = outputs.squeeze(0).permute(1, 0)
        outputs = outputs.reshape(768, 16, 16)
        return outputs


def main():
    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]

    dinov2 = DINOv2()
    inputs = dinov2.process_image(image)
    cls_token = dinov2.get_tokens(inputs)
    print(cls_token.shape)


if __name__ == "__main__":
    main()
