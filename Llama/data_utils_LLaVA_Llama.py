import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)
import torch
import os
import threading
from typing import Optional
from PIL import Image
from io import BytesIO
import requests
from huggingface_hub import hf_hub_download
nltk.download('punkt', quiet=True)

# -------------------------------
# Image cache + robust loaders
# -------------------------------

class LazyImageCache:
    """Thread-safe cache for loaded images to avoid re-downloading."""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Image.Image]:
        with self.lock:
            return self.cache.get(key)

    def put(self, key: str, image: Image.Image):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = image

    def clear(self):
        with self.lock:
            self.cache.clear()

_image_cache = LazyImageCache()

def _placeholder():
    return Image.new('RGB', (224, 224), (128, 128, 128))

def load_image_lazy(
    image_filename: str,
    repo_id: str = "liuhaotian/LLaVA-Pretrain",
    cache_dir: str = "./vqa_images_cache",
    alt_folders: Optional[list] = None
) -> Image.Image:
    """
    LLaVA-150K typically stores only image RELATIVE PATHS pointing to a separate repo.
    We try to fetch those paths from `repo_id` (default: liuhaotian/LLaVA-Pretrain).
    """
    if not isinstance(image_filename, str) or not image_filename.strip():
        return _placeholder()

    # Return from RAM cache if present
    cached_img = _image_cache.get(image_filename)
    if cached_img is not None:
        return cached_img.copy()

    # Local absolute/relative file path?
    if os.path.exists(image_filename):
        try:
            img = Image.open(image_filename).convert('RGB')
            _image_cache.put(image_filename, img)
            return img.copy()
        except Exception:
            pass

    # Check if it's a COCO image (format: 000000xxxxxx.jpg)
    base = os.path.basename(image_filename)
    if base.startswith('COCO_') or (base.startswith('000000') and base.endswith('.jpg')):
        coco_id = base.replace('COCO_train2014_', '').replace('COCO_val2014_', '').replace('.jpg', '')
        coco_urls = [
            f"http://images.cocodataset.org/train2017/{base}",
            f"http://images.cocodataset.org/val2017/{base}",
            f"http://images.cocodataset.org/train2014/COCO_train2014_{coco_id}.jpg",
            f"http://images.cocodataset.org/val2014/COCO_val2014_{coco_id}.jpg",
        ]
        
        for url in coco_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    _image_cache.put(image_filename, img)
                    return img.copy()
            except Exception:
                continue

    # Try as-is in the repo
    candidates = [image_filename]

    # Also try basename and optional alt folders
    if base != image_filename:
        candidates.append(base)

    if alt_folders:
        for folder in alt_folders:
            candidates.append(f"{folder}/{base}")

    # Common LLaVA pretrain image folders to try
    default_folders = [
        "coco/train2017",
        "coco/val2017",
        "coco/train2014",
        "coco/val2014",
        "cc_sbu_align",
        "gqa/images",
        "textcaps/images",
        "ocr_vqa/images",
        "laion/images",
        "vg/images"
    ]
    for folder in default_folders:
        candidates.append(f"{folder}/{base}")

    for cand in candidates:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=cand,
                cache_dir=cache_dir,
                repo_type="dataset"
            )
            img = Image.open(local_path)
            img.verify()
            img = Image.open(local_path).convert('RGB')
            _image_cache.put(image_filename, img)
            return img.copy()
        except Exception:
            continue

    print(f"[WARN] Could not find image {image_filename}, using placeholder")
    return _placeholder()

def load_image_from_source(image_source):
    """Robust image loading: PIL Image, URL, path, base64, bytes."""
    try:
        if isinstance(image_source, Image.Image):
            return image_source.convert('RGB') if image_source.mode != 'RGB' else image_source

        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                return image.convert('RGB') if image.mode != 'RGB' else image

            if os.path.exists(image_source):
                image = Image.open(image_source)
                return image.convert('RGB') if image.mode != 'RGB' else image

            try:
                import base64
                image_data = base64.b64decode(image_source)
                image = Image.open(BytesIO(image_data))
                return image.convert('RGB') if image.mode != 'RGB' else image
            except Exception:
                pass

        if isinstance(image_source, bytes):
            image = Image.open(BytesIO(image_source))
            return image.convert('RGB') if image.mode != 'RGB' else image

        return None

    except Exception as e:
        print(f"Failed to load image: {e}")
        return None

# -------------------------------
# LLaVA-150K adapter
# -------------------------------

def _extract_llava_qa(example):
    """
    LLaVA item format (typical):
      {
        "id": ...,
        "image": "<relative/path.jpg>" or sometimes null,
        "conversations": [
            {"from": "human", "value": "<image>\n<question ...>"},
            {"from": "gpt",   "value": "<answer ...>"},
            ...
        ]
      }

    We take the FIRST (human, gpt) pair.
    """
    image = example.get("image", None)

    conv = example.get("conversations", None)
    if not conv or len(conv) < 2:
        return None, None, None

    # Find the first human→gpt pair
    q, a = None, None
    for i in range(len(conv) - 1):
        if conv[i].get("from") == "human" and conv[i+1].get("from") == "gpt":
            q_raw = conv[i].get("value", "") or ""
            a_raw = conv[i+1].get("value", "") or ""
            # Strip LLaVA's image placeholders like "<image>", "<image>\n"
            q = q_raw.replace("<image>", "").strip()
            a = a_raw.strip()
            break

    if not q or not a:
        return None, None, None

    return image, q, a

# -------------------------------
# Llama-3.2-11B tokenization (TWO-PASS METHOD)
# -------------------------------

def tokenize_vqa(example, config, processor):
    """
    Process LLaVA-150K + generic VQA examples into Llama-3.2 Vision training inputs.
    Uses TWO-PASS tokenization:
    1. Full Conversation (Training): Question + Answer -> For Loss Calculation
    2. Prompt Only (Evaluation): Question Only -> For Generation
    """
    try:
        # 1) Try LLaVA schema first
        image_path, question, answer = _extract_llava_qa(example)

        # 2) Fallbacks for generic datasets
        if image_path is None and 'image' in example:
            image_path = example['image']
        if not question:
            question = (example.get('question', "") or "").strip()
        if not answer:
            answer = (example.get('answer', "") or "").strip()

        if not image_path or not question or not answer:
            return None

        # ---- Load image ----
        if isinstance(image_path, str) and not image_path.startswith(('http://', 'https://')):
            image = load_image_lazy(
                image_path,
                repo_id=getattr(config, "images_repo", "liuhaotian/LLaVA-Pretrain"),
                cache_dir=getattr(config, "image_cache_dir", "./vqa_images_cache"),
                alt_folders=getattr(config, "image_search_folders", None),
            )
        else:
            image = load_image_from_source(image_path)

        if image is None or not isinstance(image, Image.Image):
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ---- Build text prompts ----
        context = (example.get("context", "") or "").strip()
        if context:
            user_text = f"Context: {context}\nQuestion: {question}"
        else:
            user_text = question

        system_message = {
            "role": "system",
            "content": "You are a VQA assistant. Answer concisely in 1–2 short sentences."
        }

        # --- Messages for PASS 1 (Training): Full Conversation ---
        messages_with_answer = [
            system_message,
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        # --- Messages for PASS 2 (Evaluation): Prompt Only ---
        messages_prompt_only = [
            system_message,
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        # --- Apply Chat Templates ---
        input_text_full = processor.apply_chat_template(messages_with_answer, add_generation_prompt=False, tokenize=False)
        input_text_prompt = processor.apply_chat_template(messages_prompt_only, add_generation_prompt=True, tokenize=False)

        # --- PASS 1: Process Full Conversation (For Training) ---
        enc_full = processor(
            images=[image],
            text=[input_text_full],
            padding="max_length",
            max_length=getattr(config, "max_length", 2048),
            truncation=True,
            return_tensors="pt",
        )

        # --- PASS 2: Process Prompt Only (For Evaluation / Label Masking) ---
        enc_prompt = processor(
            images=[image],
            text=[input_text_prompt],
            padding="max_length",
            max_length=getattr(config, "max_length", 2048),
            truncation=True,
            return_tensors="pt",
        )

        # --- Build Output Dictionary ---
        
        # 1. Training Keys (Full Conversation)
        input_ids = enc_full["input_ids"].squeeze(0)
        attention_mask = enc_full["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask labels: Ignore everything in the prompt, only train on answer
        prompt_len = enc_prompt["attention_mask"].squeeze(0).sum().item()
        labels[:prompt_len] = getattr(config, "ignore_index", -100)
        
        # Also mask padding tokens
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = getattr(config, "ignore_index", -100)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # 2. Vision Tensors (Shared)
        # We extract these from enc_full. They are the same for both passes.
        if "pixel_values" in enc_full:
            out["pixel_values"] = enc_full["pixel_values"].squeeze(0)
        if "aspect_ratio_ids" in enc_full:
            out["aspect_ratio_ids"] = enc_full["aspect_ratio_ids"].squeeze(0)
        if "aspect_ratio_mask" in enc_full:
            out["aspect_ratio_mask"] = enc_full["aspect_ratio_mask"].squeeze(0)
        if "cross_attention_mask" in enc_full:
            out["cross_attention_mask"] = enc_full["cross_attention_mask"].squeeze(0)

        # 3. Evaluation Keys (Prompt Only)
        # These allow the model to generate the answer from scratch during validation
        out["eval_input_ids"] = enc_prompt["input_ids"].squeeze(0)
        out["eval_attention_mask"] = enc_prompt["attention_mask"].squeeze(0)
        if "cross_attention_mask" in enc_prompt:
            out["eval_cross_attention_mask"] = enc_prompt["cross_attention_mask"].squeeze(0)

        return out

    except Exception as e:
        print(f"Error processing example: {e}")
        import traceback; traceback.print_exc()
        return None


# def tokenize_vqa(example, config, processor):
#     """
#     Process LLaVA-150K + generic VQA examples into Qwen3-VL training inputs.
#     Uses TWO-PASS tokenization for precise label masking.
#     """
#     try:
#         # 1) Try LLaVA schema first
#         image_path, question, answer = _extract_llava_qa(example)

#         # 2) If not LLaVA, fall back to generic keys
#         if image_path is None and 'image' in example:
#             image_path = example['image']
#         if not question:
#             question = (example.get('question', "") or "").strip()
#         if not answer:
#             answer = (example.get('answer', "") or "").strip()

#         if not image_path or not question or not answer:
#             return None

#         # Load image
#         if isinstance(image_path, str) and not image_path.startswith(('http://', 'https://')):
#             image = load_image_lazy(
#                 image_path,
#                 repo_id=getattr(config, "images_repo", "liuhaotian/LLaVA-Pretrain"),
#                 cache_dir=getattr(config, "image_cache_dir", "./vqa_images_cache"),
#                 alt_folders=getattr(config, "image_search_folders", None),
#             )
#         else:
#             image = load_image_from_source(image_path)

#         if image is None or not isinstance(image, Image.Image):
#             return None
#         if image.mode != 'RGB':
#             image = image.convert('RGB')

#         # Optional context handling
#         context = (example.get('context', "") or "").strip()
#         base_prompt = f"Context: {context}\nQuestion: {question}" if context else question

#         img_token = getattr(processor, "image_token", None)

#         # Get the correct image token string from MllamaProcessor
#         prompt_content = f"{img_token}\n{base_prompt}"

#         system_message = {
#                     "role": "system",
#                     "content": "You are a VQA assistant. Answer concisely in 1-2 short sentences."
#                 }

#         # ============================================================
#         # PASS 1: Tokenize prompt ONLY (to find boundary)
#         # ============================================================
#         messages_prompt_only = [
#             system_message,
#             {"role": "user", "content": prompt_content}
#         ]

#         text_prompt_only = processor.apply_chat_template(
#             messages_prompt_only,
#             tokenize=False,
#             add_generation_prompt=True  # Ready for assistant to answer
#         )

#         enc_prompt = processor(
#             text=[text_prompt_only],
#             images=[image],
#             padding="max_length",
#             max_length=getattr(config, "max_length", 2048),
#             truncation=True,
#             return_tensors="pt"
#         )

#         # ============================================================
#         # PASS 2: Tokenize prompt + assistant answer (full conversation)
#         # ============================================================
#         messages_with_answer = [
#             system_message,
#             {"role": "user", "content": prompt_content},
#             {"role": "assistant", "content": answer}
#         ]

#         text_with_answer = processor.apply_chat_template(
#             messages_with_answer,
#             tokenize=False,
#             add_generation_prompt=False  # Complete conversation
#         )

#         enc_full = processor(
#             text=[text_with_answer],
#             images=[image],
#             padding="max_length",
#             max_length=getattr(config, "max_length", 2048),
#             truncation=True,
#             return_tensors="pt"
#         )

#         # ============================================================
#         # LABELS: Mask prompt, supervise only assistant's answer
#         # ============================================================
#         input_ids = enc_full["input_ids"].squeeze(0)
#         attention_mask = enc_full["attention_mask"].squeeze(0)

#         # Clone full sequence for labels
#         labels = input_ids.clone()
#         labels[:] = getattr(config, "ignore_index", -100)

#         # Get exact prompt length from first pass
#         # prompt_len = enc_prompt["input_ids"].shape[-1]
#         prompt_len = enc_prompt["attention_mask"].squeeze(0).sum().item()

#         # Unmask only the assistant's response (everything after prompt)
#         # Unmask/copy assistant tokens
#         labels[prompt_len:] = input_ids[prompt_len:]

#         # Both sequences are padded to same max_length, so boundaries align
#         labels[:prompt_len] = getattr(config, "ignore_index", -100)

#         # Also mask padding tokens to be safe
#         pad_id = processor.tokenizer.pad_token_id
#         if pad_id is not None:
#             labels[input_ids == pad_id] = getattr(config, "ignore_index", -100)

#         # Prepare output
#         output = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels
#         }

#         # ============================================================
#         # Add vision tensors (pixel_values, image_sizes)
#         # ============================================================

#         if "pixel_values" in enc_full:
#             pv = enc_full["pixel_values"]
#             if isinstance(pv, list) and len(pv) > 0 and isinstance(pv[0], torch.Tensor):
#                 pv = pv[0] # Get the first (and only) image tensor
#             elif isinstance(pv, torch.Tensor) and pv.dim() >= 2 and pv.shape[0] == 1:
#                 pv = pv.squeeze(0)
#             output["pixel_values"] = pv

#         # Extract image_sizes, which is required by Llama 3.2 Vision
#         if "image_sizes" in enc_full:
#             img_sz = enc_full["image_sizes"]
#             if isinstance(img_sz, list) and len(img_sz) > 0 and isinstance(img_sz[0], torch.Tensor):
#                 img_sz = img_sz[0] # Get the first (and only) image_sizes tensor
#             elif isinstance(img_sz, torch.Tensor) and img_sz.dim() == 2 and img_sz.shape[0] == 1:
#                 img_sz = img_sz.squeeze(0) # Squeeze batch dim    

#             # Ensure it's a 1D tensor of shape [2] (H, W)
#             if isinstance(img_sz, torch.Tensor) and img_sz.dim() == 1 and img_sz.shape[0] == 2:
#                 output["image_sizes"] = img_sz

#         # Final validation
#         for k in ["input_ids", "attention_mask", "labels"]:
#             if k in output and not isinstance(output[k], torch.Tensor):
#                 return None


#         if "pixel_values" not in output or "image_sizes" not in output:
#              return None # We must have both vision tensors
             
#         if "image_sizes" in output:
#             igt = output["image_sizes"]
#             if not (isinstance(igt, torch.Tensor) and igt.dim() == 1 and igt.shape[0] == 2):
#                 print(f"[WARN] Invalid image_sizes shape: {igt.shape}, skipping")
#                 return None

#         return output

#     except Exception as e:
#         print(f"Error processing example: {e}")
#         import traceback; traceback.print_exc()
#         return None

# -------------------------------
# Collator (Adapted for Llama 3.2 Vision)
# -------------------------------

def collate_fn(batch, config, processor):
    """
    Custom collator for Llama 3.2 Vision inputs.
    Robustly handles cases where the dataset yields Lists instead of Tensors.
    """
    # Filter out bad samples
    batch = [b for b in batch if b is not None and isinstance(b, dict) and 'input_ids' in b]

    if len(batch) == 0:
        return _create_dummy_batch(config, processor)

    output = {}

    # 1. Handle Text Tensors (input_ids, attention_mask, labels)
    #    We convert lists to tensors if necessary
    text_keys = ['input_ids', 'attention_mask', 'labels']
    for key in text_keys:
        # Prepare the list of tensors for padding
        tensor_list = []
        for b in batch:
            item = b[key]
            if not isinstance(item, torch.Tensor):
                item = torch.tensor(item, dtype=torch.long)
            tensor_list.append(item)
        
        # Pad sequence
        padding_val = getattr(config, "ignore_index", -100) if key == 'labels' else (processor.tokenizer.pad_token_id if key == 'input_ids' else 0)
        output[key] = torch.nn.utils.rnn.pad_sequence(
            tensor_list,
            batch_first=True,
            padding_value=padding_val
        )

    # 2. Handle Vision Tensors (pixel_values, masks, etc.)
    vision_keys = ["pixel_values", "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"]
    
    for key in vision_keys:
        # Check if this key exists in the first sample
        if key in batch[0] and batch[0][key] is not None:
            items_to_stack = []
            
            for b in batch:
                val = b[key]
                # CRITICAL FIX: Convert List -> Tensor
                if not isinstance(val, torch.Tensor):
                    # pixel_values need to be float/bfloat, others are long (int)
                    if key == "pixel_values":
                        val = torch.tensor(val, dtype=torch.float32) # float32 is safest, model will cast to bf16
                    else:
                        val = torch.tensor(val, dtype=torch.long)
                items_to_stack.append(val)
            
            # Stack them (Batch Size 1 works fine here; Batch Size > 1 requires same aspect ratio)
            try:
                output[key] = torch.stack(items_to_stack, dim=0)
            except Exception as e:
                print(f"Error stacking {key}: {e}")
                # Don't crash, just omit the key (will treat as text-only for this batch)
                pass

    return output

def _create_dummy_batch(config, processor):
    """Helper to create a dummy batch if all data is filtered out."""
    print("WARNING: Creating dummy batch")
    max_len = getattr(config, "max_length", 2048)
    return {
        'input_ids': torch.zeros((1, max_len), dtype=torch.long),
        'attention_mask': torch.zeros((1, max_len), dtype=torch.long),
        'labels': torch.full((1, max_len), getattr(config, "ignore_index", -100), dtype=torch.long)
    }

        # # pixel_values (optional)
        # if 'pixel_values' in batch[0] and batch[0]['pixel_values'] is not None:
        #     pixel_values_list = []
        #     for b in batch:
        #         pv = b['pixel_values']
        #         # Ensure each item is a tensor
        #         if isinstance(pv, list):
        #             if len(pv) > 0 and isinstance(pv[0], torch.Tensor):
        #                 pv = pv[0]
        #             else:
        #                 continue
        #         if isinstance(pv, torch.Tensor):
        #             pixel_values_list.append(pv)
            
        #     if len(pixel_values_list) > 0:
        #         try:
        #             output['pixel_values'] = torch.stack(pixel_values_list, dim=0)
        #         except RuntimeError as e:
        #             print(f"Error stacking pixel_values: {e}")
        #             # Fallback: pad to same shape
        #             max_shape = [max(pv.shape[i] for pv in pixel_values_list) for i in range(pixel_values_list[0].ndim)]
        #             padded_pvs = []
        #             for pv in pixel_values_list:
        #                 if list(pv.shape) != max_shape:
        #                     pad_sizes = [(0, max_shape[i] - pv.shape[i]) for i in range(len(max_shape))]
        #                     pad_sizes = [item for sublist in reversed(pad_sizes) for item in sublist]
        #                     pv = torch.nn.functional.pad(pv, pad_sizes)
        #                 padded_pvs.append(pv)
        #             output['pixel_values'] = torch.stack(padded_pvs, dim=0)

        # # This is a [B, 2] tensor (Batch_size, [H, W])

        # if 'image_sizes' in batch[0] and batch[0]['image_sizes'] is not None:
        #     # Force cast to tensor just in case tokenization missed it
        #     image_sizes_list = [
        #         b['image_sizes'] if isinstance(b['image_sizes'], torch.Tensor) 
        #         else torch.tensor(b['image_sizes'], dtype=torch.long) 
        #         for b in batch
        #     ]
        #     if len(image_sizes_list) > 0:
        #         output['image_sizes'] = torch.stack(image_sizes_list, dim=0)

        # return output

    # except Exception as e:
    #     print(f"ERROR in collate_fn: {e}")
    #     import traceback; traceback.print_exc()
    #     dummy_input_ids = torch.zeros((1, getattr(config, "max_length", 2048)), dtype=torch.long)
    #     dummy_attention_mask = torch.zeros((1, getattr(config, "max_length", 2048)), dtype=torch.long)
    #     dummy_labels = torch.full((1, getattr(config, "max_length", 2048)), getattr(config, "ignore_index", -100), dtype=torch.long)
    #     return {'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask, 'labels': dummy_labels}