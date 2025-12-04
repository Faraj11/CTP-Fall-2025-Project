import torch
from PIL import Image
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel


class ImageCaptioner:
    """
    Singleton class for generating image captions using Hugging Face's
    VisionEncoderDecoderModel (ViT-GPT2 image captioning).
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageCaptioner, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the model, tokenizer, and feature extractor."""
        self.device = "cpu"
        self.encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        self.decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        self.model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"

        print("[ImageCaptioner] Loading feature extractor...")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.encoder_checkpoint)
        print("[ImageCaptioner] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.decoder_checkpoint)
        print("[ImageCaptioner] Loading model...")
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_checkpoint).to(self.device)
        print("[ImageCaptioner] Model loaded successfully.")

    def generate_caption(self, image: Image.Image, max_length: int = 64, num_beams: int = 4) -> str:
        """
        Generate a caption for the given image.

        Args:
            image: PIL Image object
            max_length: Maximum length of the generated caption
            num_beams: Number of beams for beam search

        Returns:
            Generated caption as a string
        """
        if image is None:
            return "No image provided."

        # Convert image to RGB if needed
        image = image.convert("RGB")
        
        # Extract features
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.to(self.device)

        # Generate caption
        caption_ids = self.model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams
        )[0]

        # Decode and clean the caption
        clean_text = lambda x: x.replace("<|endoftext|>", "").split("\n")[0]
        caption_text = clean_text(self.tokenizer.decode(caption_ids))
        
        return caption_text
