import re
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptioner:
    """
    Singleton class for generating detailed, accurate image captions using BLIP-2.
    BLIP-2 provides superior image understanding and generates more detailed,
    food-specific captions compared to ViT-GPT2.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageCaptioner, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance.model = None
            cls._instance.processor = None
            cls._instance.device = None
            cls._instance.model_name = None
        return cls._instance

    def _initialize(self):
        """Initialize the BLIP model, processor, and tokenizer (lazy loading)."""
        if self._initialized and self.model is not None:
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Using BLIP (not BLIP-2) for better CPU compatibility and faster inference
        # BLIP still provides much better results than ViT-GPT2
        self.model_name = "Salesforce/blip-image-captioning-base"
        
        print("[ImageCaptioner] Loading BLIP processor and model...")
        print(f"[ImageCaptioner] Using device: {self.device}")
        
        try:
            # Clear any cached memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load with low_cpu_mem_usage to reduce memory footprint
            self.processor = BlipProcessor.from_pretrained(
                self.model_name,
                use_fast=True  # Use fast tokenizer
            )
            # Use device_map="auto" for better memory management if available
            try:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,  # Reduce memory usage
                    torch_dtype=torch.float32  # Use float32 instead of float16 for compatibility
                )
                self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode
                
                # Enable memory efficient attention if available
                if hasattr(self.model, 'config'):
                    try:
                        self.model.config.use_cache = False  # Disable cache to save memory
                    except:
                        pass
            except MemoryError:
                # Try with even lower memory settings
                raise MemoryError("Not enough memory to load model. Consider upgrading server.")
            
            # Clear cache after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._initialized = True
            print("[ImageCaptioner] BLIP model loaded successfully.")
        except MemoryError as e:
            print(f"[ImageCaptioner] Memory error loading BLIP model: {e}")
            print("[ImageCaptioner] This may be due to limited memory on the server.")
            self.model = None
            self.processor = None
            self._initialized = False
            raise RuntimeError("Insufficient memory to load image captioning model. Please upgrade your server plan.")
        except Exception as e:
            print(f"[ImageCaptioner] Error loading BLIP model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None
            self._initialized = False
            raise RuntimeError(f"Failed to load BLIP model: {str(e)}")

    def generate_caption(self, image: Image.Image, max_length: int = 64, num_beams: int = 5) -> str:
        """
        Generate a detailed, accurate caption for the given food image.
        
        Uses prompt engineering to get food-specific, detailed descriptions.

        Args:
            image: PIL Image object
            max_length: Maximum length of the generated caption (default 64, increased for detail)
            num_beams: Number of beams for beam search (default 5 for better quality)

        Returns:
            Generated caption as a string with detailed food description
        """
        if image is None:
            return "No image provided."
        
        # Lazy load model on first use (not at startup to avoid memory issues)
        if not self._initialized or self.model is None:
            try:
                self._initialize()
            except RuntimeError as e:
                return f"Unable to load image captioning model: {str(e)}"
            except Exception as e:
                return f"Error initializing image captioning: {str(e)}"
        
        # Double-check model is loaded
        if self.model is None or self.processor is None:
            return "Image captioning model not available."

        # Convert image to RGB if needed
        image = image.convert("RGB")
        
        try:
            # Clear cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use unconditional generation (no prompt) to avoid prompt text in output
            # BLIP generates better captions without prompts for food images
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Get pad_token_id safely
            pad_token_id = None
            if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'pad_token_id'):
                pad_token_id = self.processor.tokenizer.pad_token_id
            elif hasattr(self.processor, 'pad_token_id'):
                pad_token_id = self.processor.pad_token_id
            
            # If pad_token_id is None, use eos_token_id as fallback
            if pad_token_id is None:
                if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'eos_token_id'):
                    pad_token_id = self.processor.tokenizer.eos_token_id
                elif hasattr(self.processor, 'eos_token_id'):
                    pad_token_id = self.processor.eos_token_id
            
            # Get eos_token_id for proper stopping
            eos_token_id = None
            if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'eos_token_id'):
                eos_token_id = self.processor.tokenizer.eos_token_id
            elif hasattr(self.processor, 'eos_token_id'):
                eos_token_id = self.processor.eos_token_id
            
            # Prepare generation kwargs with improved parameters to reduce typos and cut-off words
            generation_kwargs = {
                "max_length": max_length + 10,  # Add buffer to prevent cut-off words
                "num_beams": num_beams,
                "num_return_sequences": 1,
                "temperature": 0.6,  # Lower temperature for more accurate, less creative text
                "do_sample": True,  # Enable sampling but with lower temperature
                "repetition_penalty": 1.3,  # Moderate penalty to reduce repetition without over-correction
                "length_penalty": 1.0,  # Neutral length penalty to avoid cutting off
                "early_stopping": False,  # Disable early stopping to prevent cut-off words
                "no_repeat_ngram_size": 3,  # Prevent 3-gram repetition
            }
            
            # Add token IDs if available
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
            
            # Generate caption with improved parameters for better quality
            with torch.no_grad():
                out = self.model.generate(**inputs, **generation_kwargs)
            
            # Decode the generated caption with proper handling
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Fix incomplete words at the end (common issue with cut-off)
            caption = self._fix_incomplete_words(caption)
            
            # Post-process to enhance food-specific details and remove unwanted text
            caption = self._enhance_food_description(caption)
            
            # Ensure we have a valid caption
            if not caption or len(caption.strip()) == 0:
                return "Unable to generate caption for this image."
            
            return caption.strip()
            
        except MemoryError as e:
            print(f"[ImageCaptioner] Memory error generating caption: {e}")
            # Clear cache and raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise MemoryError("Insufficient memory to process image. Please try a smaller image or upgrade server resources.")
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                print(f"[ImageCaptioner] CUDA/GPU memory error: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise MemoryError("GPU memory error. Please try again or use a smaller image.")
            raise  # Re-raise other RuntimeErrors
        except Exception as e:
            print(f"[ImageCaptioner] Error generating caption: {e}")
            import traceback
            print("[ImageCaptioner] Full traceback:")
            traceback.print_exc()
            # Try a simpler fallback approach
            try:
                print("[ImageCaptioner] Attempting fallback generation...")
                # Clear cache before fallback
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                caption = self._enhance_food_description(caption)
                if caption and len(caption.strip()) > 0:
                    return caption.strip()
            except MemoryError as e2:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise MemoryError("Insufficient memory for image processing.")
            except Exception as e2:
                print(f"[ImageCaptioner] Fallback also failed: {e2}")
            
            return "Unable to generate caption for this image."

    def _enhance_food_description(self, caption: str) -> str:
        """
        Post-process caption to enhance food-specific details and accuracy.
        Removes unwanted prompt text, generic prefixes, and improves food descriptions.
        """
        caption = caption.strip()
        
        # Remove prompt text that might have been included in output
        prompt_patterns = [
            "a detailed description of this food dish",
            "a detailed description of this food",
            "detailed description of this food dish",
            "detailed description of this food",
            "description of this food dish",
            "description of this food",
        ]
        caption_lower = caption.lower()
        for pattern in prompt_patterns:
            if pattern in caption_lower:
                # Remove the pattern and clean up (case-insensitive)
                caption = re.sub(re.escape(pattern), "", caption, flags=re.IGNORECASE)
                caption = caption.strip()
                # Remove leading/trailing commas and extra spaces
                caption = re.sub(r'^[,.\s]+', '', caption)
                caption = re.sub(r'[,.\s]+$', '', caption)
                caption = re.sub(r'\s+', ' ', caption)
                break
        
        # Remove generic prefixes that don't add value
        generic_prefixes = [
            "a photo of ",
            "an image of ",
            "a picture of ",
            "a photograph of ",
            "photo of ",
            "image of ",
            "picture of ",
            "photograph of ",
            "this is ",
            "this shows ",
            "this image shows ",
            "this photo shows ",
        ]
        caption_lower = caption.lower()
        for prefix in generic_prefixes:
            if caption_lower.startswith(prefix):
                caption = caption[len(prefix):].strip()
                # Capitalize first letter
                if caption:
                    caption = caption[0].upper() + caption[1:] if len(caption) > 1 else caption.upper()
                break
        
        # Remove repetitive phrases (e.g., "with maple syrup and maple syrup")
        # First, handle duplicate phrases with "and" or commas
        # Pattern: "X and X" or "X, X" (case-insensitive)
        caption = re.sub(r'\b(\w+(?:\s+\w+)*)\s+and\s+\1\b', r'\1', caption, flags=re.IGNORECASE)
        caption = re.sub(r'\b(\w+(?:\s+\w+)*),\s*\1\b', r'\1', caption, flags=re.IGNORECASE)
        caption = re.sub(r'\b(\w+(?:\s+\w+)*)\s+with\s+\1\b', r'with \1', caption, flags=re.IGNORECASE)
        
        # Remove duplicate consecutive words (e.g., "french fries french fries")
        words = caption.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            # Check if current word and next few words form a duplicate pattern
            if i + 1 < len(words) and words[i].lower() == words[i + 1].lower():
                # Skip the duplicate
                cleaned_words.append(words[i])
                i += 2
                # Skip any additional duplicates
                while i < len(words) and words[i].lower() == cleaned_words[-1].lower():
                    i += 1
            else:
                cleaned_words.append(words[i])
                i += 1
        caption = ' '.join(cleaned_words)
        
        # Clean up extra spaces and punctuation
        caption = re.sub(r'\s+', ' ', caption)
        caption = re.sub(r'\s*,\s*,+', ',', caption)  # Remove multiple commas
        caption = re.sub(r'^[,.\s]+', '', caption)  # Remove leading punctuation
        caption = re.sub(r'[,.\s]+$', '', caption)  # Remove trailing punctuation
        
        # Capitalize first letter if needed
        if caption and caption[0].islower():
            caption = caption[0].upper() + caption[1:]
        
        # Final cleanup and spell checking for common food terms
        caption = self._fix_common_typos(caption)
        
        return caption.strip()
    
    def _fix_incomplete_words(self, caption: str) -> str:
        """
        Fix incomplete words that may have been cut off during generation.
        Common issue: words ending abruptly like "spaghett" instead of "spaghetti"
        """
        if not caption:
            return caption
        
        # Common food word completions
        word_completions = {
            'spaghett': 'spaghetti',
            'meatbal': 'meatball',
            'hamburge': 'hamburger',
            'burg': 'burger',
            'pancak': 'pancake',
            'blueberr': 'blueberry',
            'syru': 'syrup',
            'sau': 'sauce',
            'marinar': 'marinara',
            'parmesa': 'parmesan',
            'mozzarell': 'mozzarella',
            'french fri': 'french fries',
            'frie': 'fries',
        }
        
        words = caption.split()
        fixed_words = []
        
        for word in words:
            # Remove trailing punctuation temporarily
            trailing_punct = ''
            if word and word[-1] in '.,!?;:':
                trailing_punct = word[-1]
                word_clean = word[:-1]
            else:
                word_clean = word
            
            # Check if word needs completion
            word_lower = word_clean.lower()
            if word_lower in word_completions:
                # Replace with completed version, preserving capitalization
                if word_clean[0].isupper():
                    fixed_word = word_completions[word_lower].capitalize()
                else:
                    fixed_word = word_completions[word_lower]
                fixed_words.append(fixed_word + trailing_punct)
            else:
                fixed_words.append(word)
        
        return ' '.join(fixed_words)
    
    def _fix_common_typos(self, caption: str) -> str:
        """
        Fix common typos in food-related captions.
        """
        if not caption:
            return caption
        
        # Common food-related typos and corrections
        typo_corrections = {
            # Common misspellings
            r'\bspagetti\b': 'spaghetti',
            r'\bspaghett\b': 'spaghetti',
            r'\bmeatbal\b': 'meatball',
            r'\bhamburge\b': 'hamburger',
            r'\bburg\b': 'burger',
            r'\bpancak\b': 'pancake',
            r'\bblueberr\b': 'blueberry',
            r'\bsyru\b': 'syrup',
            r'\bmarinar\b': 'marinara',
            r'\bparmesa\b': 'parmesan',
            r'\bmozzarell\b': 'mozzarella',
            r'\bfrench fri\b': 'french fries',
            r'\bfrie\b': 'fries',
            # Common word boundary issues
            r'\bfrenchfries\b': 'french fries',
            r'\bfrench-fries\b': 'french fries',
            r'\bmeat-balls\b': 'meatballs',
            r'\bblueberrypancakes\b': 'blueberry pancakes',
            r'\bmaple-syrup\b': 'maple syrup',
            r'\btomatosauce\b': 'tomato sauce',
            r'\btomato-sauce\b': 'tomato sauce',
        }
        
        caption_lower = caption.lower()
        for typo_pattern, correction in typo_corrections.items():
            if re.search(typo_pattern, caption_lower, re.IGNORECASE):
                # Replace with case-aware correction
                matches = list(re.finditer(typo_pattern, caption, re.IGNORECASE))
                for match in reversed(matches):  # Reverse to maintain indices
                    start, end = match.span()
                    matched_text = caption[start:end]
                    # Preserve capitalization
                    if matched_text[0].isupper():
                        replacement = correction.capitalize()
                    else:
                        replacement = correction
                    caption = caption[:start] + replacement + caption[end:]
        
        # Fix common spacing issues
        caption = re.sub(r'([a-z])([A-Z])', r'\1 \2', caption)  # Add space between camelCase
        caption = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', caption)  # Add space before numbers
        caption = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', caption)  # Add space after numbers
        
        # Fix double spaces
        caption = re.sub(r'\s+', ' ', caption)
        
        return caption
