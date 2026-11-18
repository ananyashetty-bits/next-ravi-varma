from functools import lru_cache
from transformers import CLIPTokenizer
import json, os

CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "m3.json")))   # works from any directory

@lru_cache(1)
def _tok(): return CLIPTokenizer.from_pretrained(CFG["base_model"], subfolder="tokenizer")

def n_tokens(text): return len(_tok()(text, add_special_tokens=False).input_ids)   # limit = config max_prompt_tokens