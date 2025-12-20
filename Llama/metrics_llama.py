import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)

from transformers import EvalPrediction

nltk.download('punkt', quiet=True)

import re
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from sklearn.metrics import f1_score


def make_compute_metrics(processor):
    """
    Creates metrics computation function for VQA evaluation.
    Fixed to properly decode labels by extracting only answer tokens.
    """
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def extract_answer(text):
        """Extract clean answer from Qwen3-VL generation."""
        # Remove special tokens
        text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        text = text.replace("<|vision_start|>", "").replace("<|vision_end|>", "")

        # If there's an "assistant" marker, take what comes after
        if "assistant" in text.lower():
            parts = re.split(r'assistant\s*', text, flags=re.IGNORECASE)
            if len(parts) > 1:
                text = parts[-1]

        # Clean up
        text = text.strip()

        # For VQA, often the answer is in the first sentence
        sentences = re.split(r'[.!?\n]+', text)
        if sentences and len(sentences[0].split()) <= 50:  # Short answer expected
            return sentences[0].strip()

        return text

    def normalize(s):
        """VQA normalization."""
        s = re.sub(r"\s+", " ", re.sub(r"([.!?,;])", r" \1", s.lower().strip()))
        s = re.sub(r" n't ", " not ", s)
        s = re.sub(r" 's ", " is ", s)
        s = re.sub(r" 're ", " are ", s)
        s = re.sub(r" 've ", " have ", s)
        s = re.sub(r" 'll ", " will ", s)
        s = re.sub(r" 'd ", " would ", s)
        return s.strip()

    def compute_vqa_metrics(p: EvalPrediction):
        """
        Compute VQA metrics: ROUGE-L, METEOR, BERTScore.
        Minimal, robust fixes:
          - handle logits/tuple predictions
          - avoid clipping (map OOR -> PAD)
          - trim at EOS/PAD before decode
          - unicode normalize
        """
        import unicodedata

        # --- helpers ---------------------------------------------------------
        def to_int_ids(arr):
            """Map predictions to int ids (handle logits / tuple)."""
            if isinstance(arr, (list, tuple)):
                arr = arr[0]  # common case: (logits, ...) or (ids,)
            arr = np.asarray(arr)
            # If logits: take argmax on last axis
            if arr.ndim == 3:
                arr = arr.argmax(-1)
            return arr.astype(np.int64)

        def trim_to_answer_span(ids_row, pad_id, eos_id):
            """Cut at first EOS or PAD; drop leading pads."""
            ids = ids_row.tolist()
            # find first eos/pad
            cut = len(ids)
            for i, t in enumerate(ids):
                if (eos_id is not None and t == eos_id) or (pad_id is not None and t == pad_id):
                    cut = i
                    break
            # drop leading pads
            start = 0
            if pad_id is not None:
                while start < cut and ids[start] == pad_id:
                    start += 1
            return np.asarray(ids[start:cut], dtype=np.int64)

        # --- tokenizer things ------------------------------------------------
        pad_token_id = processor.tokenizer.pad_token_id
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
        vocab_size = processor.tokenizer.vocab_size
        ignore_index = -100

        # --- predictions: ids not logits, trimmed at EOS/PAD -----------------
        pred_ids = to_int_ids(p.predictions)

        # Replace out-of-range ids with PAD (avoid weird chars from clipping)
        if vocab_size is not None:
            invalid_mask = (pred_ids < 0) | (pred_ids >= vocab_size)
            if pad_token_id is None:
                # Fall back to 0 if tokenizer lacks pad (rare)
                pred_ids[invalid_mask] = 0
            else:
                pred_ids[invalid_mask] = pad_token_id

        # Per-sample trim
        trimmed_preds = [trim_to_answer_span(row, pad_token_id, eos_token_id) for row in pred_ids]

        # --- labels: you already extract only the answer tokens ---------------
        labels = np.asarray(p.label_ids, dtype=np.int64)
        label_texts = []
        for label_seq in labels:
            valid_mask = (label_seq != ignore_index) & (label_seq != pad_token_id)
            answer_tokens = label_seq[valid_mask]
            if answer_tokens.size > 0:
                try:
                    txt = processor.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                except Exception as e:
                    print(f"[Metrics] Warning: Failed to decode answer tokens: {e}")
                    txt = ""
            else:
                txt = ""
            label_texts.append(txt)

        # --- decode predictions ----------------------------------------------
        try:
            pred_texts_raw = processor.tokenizer.batch_decode(trimmed_preds, skip_special_tokens=True)
        except Exception as e:
            print(f"[Metrics] Decode error for predictions: {e}")
            return {"rouge_l": 0.0, "meteor": 0.0, "bertscore_f1": 0.0}

        # Your existing cleaner
        pred_texts = [extract_answer(t) for t in pred_texts_raw]

        # --- normalize (incl. Unicode) ---------------------------------------
        def uclean(s):  # unicode + your normalize
            s = unicodedata.normalize("NFKC", s or "")
            return normalize(s)

        pred_norm = [uclean(t) for t in pred_texts]
        label_norm = [uclean(t) for t in label_texts]

        # --- debug samples ----------------------------------------------------
        print("\n" + "="*60)
        print("EVALUATION SAMPLES")
        print("="*60)
        for i in range(min(3, len(pred_norm))):
            print(f"\n[Sample {i+1}]")
            print(f"Generated: {pred_texts[i][:100]}")
            print(f"Reference: {label_texts[i][:100]}")
        print("="*60 + "\n")

        # --- ROUGE-L ---------------------------------------------------------
        rouge_ls = []
        for ptxt, ltxt in zip(pred_norm, label_norm):
            if ptxt and ltxt:
                try:
                    rouge_ls.append(scorer.score(ltxt, ptxt)['rougeL'].fmeasure)
                except Exception:
                    pass
        rouge_l = float(np.mean(rouge_ls)) if rouge_ls else 0.0

        # --- METEOR ----------------------------------------------------------
        meteors = []
        for ptxt, ltxt in zip(pred_norm, label_norm):
            if ptxt and ltxt:
                try:
                    meteors.append(meteor_score([ltxt.split()], ptxt.split()))
                except Exception:
                    pass
        meteor = float(np.mean(meteors)) if meteors else 0.0

        # --- BERTScore -------------------------------------------------------
        valid_pairs = [(p_, l_) for p_, l_ in zip(pred_norm, label_norm) if p_ and l_]
        if valid_pairs:
            valid_preds = [p_ for p_, _ in valid_pairs]
            valid_labels = [l_ for _, l_ in valid_pairs]
            try:
                P, R, F1 = bert_score(
                    valid_preds, valid_labels,
                    lang="en",
                    model_type="bert-base-uncased",
                    rescale_with_baseline=True,
                    verbose=False
                )
                bertscore_f1 = float(F1.mean().item())
            except Exception as e:
                print(f"[Metrics] BERTScore computation error: {e}")
                bertscore_f1 = 0.0
        else:
            bertscore_f1 = 0.0

        # --- coverage stats ---------------------------------------------------
        empty_preds = sum(1 for s in pred_norm if not s)
        empty_labels = sum(1 for s in label_norm if not s)
        print(f"\nMetrics Statistics:")
        print(f"  Total samples: {len(pred_norm)}")
        print(f"  Empty predictions: {empty_preds}")
        print(f"  Empty references: {empty_labels}")
        print(f"  Valid pairs for scoring: {len(valid_pairs)}")

        return {"rouge_l": rouge_l, "meteor": meteor, "bertscore_f1": bertscore_f1}

    return compute_vqa_metrics