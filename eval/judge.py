"""Checklist judge (Member B).

A small vision-language model answers each yes/no checklist question about ONE image.
It sees only the image and one question: never the prompt, the condition or other questions.
Score = P(yes) from the token probabilities (graded, not a noisy 0/1).

Usage:
    from eval.judge import Judge
    j = Judge(cache_path="/content/drive/MyDrive/m3/cache/judge_cache.json")
    score, detail = j.checklist("img.png", scene["checklist"])
    j.save()
"""
import hashlib
import json
import os

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"   # pinned on Day 1; change only via config PR and say so in EVAL_REPORT.md
SUFFIX = " Answer with only Yes or No."

_SHA = {}


def sha256_file(path):
    if path not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


class Judge:
    def __init__(self, model_id=MODEL_ID, cache_path=None):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.model_id = model_id
        # 512x512 images stay at native resolution (no upscaling)
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=128 * 28 * 28, max_pixels=512 * 512)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda").eval()
        tok = self.proc.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")})
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            self.cache = json.load(open(cache_path))
        self._new = 0

    @torch.no_grad()
    def p_yes(self, image_path, question):
        key = f"{sha256_file(image_path)}|{self.model_id}|{question}"
        if key in self.cache:
            return self.cache[key]
        img = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + SUFFIX}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        logits = self.model(**inp).logits[0, -1].float()
        p = torch.sigmoid(torch.logsumexp(logits[self.yes], 0) - torch.logsumexp(logits[self.no], 0)).item()
        self.cache[key] = p
        self._new += 1
        if self._new % 50 == 0:
            self.save()
        return p

    def checklist(self, image_path, checklist):
        """Weighted mean P(yes) over the scene's questions, plus per-question detail."""
        detail = {c["id"]: self.p_yes(image_path, c["q"]) for c in checklist}
        w = sum(c.get("weight", 1.0) for c in checklist)
        score = sum(detail[c["id"]] * c.get("weight", 1.0) for c in checklist) / w
        return score, detail

    def save(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            json.dump(self.cache, open(self.cache_path, "w"))
