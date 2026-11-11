"""train_v3 = train_v1 with character names added to captions: 'rrvarma style, <Name>, <Name2>, <v1 body>'.
Same files as train_v1 (asserted). Names only for epic/mythological characters present in the image's
scene_metadata (historical portraits and non-epic figures excluded). Run from the repo root."""
import hashlib, json, os, random, shutil, sys
sys.path.insert(0, os.getcwd())
from common.tokens import n_tokens

MD = "/content/drive/MyDrive/m2"
V1 = f"{MD}/dataset/train_v1"
OUT = "/content/drive/MyDrive/m3/dataset/train_v3"
TOKEN_LIMIT = 75

EXCLUDE = {  # historical portraits / not epic. keys as in train/coverage_table.json
 "kerala varma valiya koil tampuran", "mahaprabha thampuratti of mavelikkara", "marthanda varma",
 "rani bharani thirunal lakshmi bayi", "shivaji maharaj", "rai pannalal mehta",
 "bharani thirunal rani parvathi bayi", "adi shankaracharya", "raja raja varma",
 "ayilyam thirunal rama varma", "yuvaraj kanterava narasimharaja wadiyar", "moolam thirunal rama varma",
 "justice t. muthuswamy iyer", "senior rani lakshmi bayi", "krishnaraja wadiyar iv", "jayalakshmanni",
 "krishnarajammanni", "mir mehboob ali khan, asaf jah vi", "maharana fateh singh of mewar",
 "shungrasoobyer avergal", "princess tarabai", "seth lachman das", "nayar lady", "judith",
 "kadambari", "malati", "vasantika", "varini", "tara devi", "gopichandra"}

table = json.load(open("train/coverage_table.json"))["characters"]
cap = json.load(open(f"{MD}/m1/final_captions.json"))
kb = json.load(open("kb/kb.json", encoding="utf-8"))["characters"]
display = {k: kb[k]["canonical"] if k in kb else k.title() for k in table}
raw2key = {raw.strip().lower(): k for k, t in table.items() for raw in t["raw_variants"]}

rows = [json.loads(l) for l in open(f"{V1}/metadata.jsonl")]
os.makedirs(OUT, exist_ok=True)
changed, dropped_names, max_tok, per_char = 0, [], 0, {}
with open(f"{OUT}/metadata.jsonl", "w") as f:
    for r in rows:
        stem = os.path.splitext(r["file_name"])[0]
        shutil.copy(f"{V1}/{r['file_name']}", f"{OUT}/{r['file_name']}")
        names = []
        for c in cap[stem + ".png"]["scene_metadata"]["characters"]:
            k = raw2key.get(c.strip().lower())
            if k and k not in EXCLUDE and display[k] not in names:
                names.append(display[k])
        body = r["text"].replace("rrvarma style, ", "", 1)
        def build(ns): return "rrvarma style, " + (", ".join(ns) + ", " if ns else "") + body
        text = build(names)
        while names and n_tokens(text) > TOKEN_LIMIT:
            dropped_names.append((stem, names.pop())); text = build(names)
        assert text.count("rrvarma style") == 1 and text.startswith("rrvarma style")
        max_tok = max(max_tok, n_tokens(text))
        for n in names: per_char[n] = per_char.get(n, 0) + 1
        changed += bool(names)
        f.write(json.dumps({"file_name": r["file_name"], "text": text}) + "\n")

new = [json.loads(l) for l in open(f"{OUT}/metadata.jsonl")]
assert [r["file_name"] for r in new] == [r["file_name"] for r in rows], "file list differs from train_v1"
sha = hashlib.sha256(open(f"{OUT}/metadata.jsonl", "rb").read()).hexdigest()
open("train/train_v3_sha256.txt", "w").write(sha + "\n")
print(f"{len(new)} captions | {changed} with names | {len(per_char)} distinct names | max tokens {max_tok} | sha256 {sha[:16]}")
print("names dropped to fit the token limit:", dropped_names or "none")
print("included:", sorted(per_char.items(), key=lambda kv: -kv[1]))
print("excluded (present in data):", sorted(k for k in EXCLUDE if k in table))
random.seed(0)
print("\n--- 15 random captions to spot-read ---")
for r in random.sample(new, 15):
    print(r["file_name"], "\n   ", r["text"])
