"""Coverage audit: which characters appear in the training images (train_v1 = 147 images).
Reads M1 scene_metadata, drops the images removed in DATA_FROZEN.json, counts characters.
Writes train/coverage_table.json (used later to build the KB)."""
import json, collections

MD = "/content/drive/MyDrive/m2"
cap = json.load(open(f"{MD}/m1/final_captions.json"))
frozen = json.load(open(f"{MD}/dataset/DATA_FROZEN.json"))
dropped = set(frozen.get("dropped", []))
stem = lambda k: k.rsplit(".", 1)[0]

kept = {k: v for k, v in cap.items() if stem(k) not in dropped}
assert len(kept) == frozen["n_train"], f"kept {len(kept)} != n_train {frozen['n_train']}"

# Grow this as you see spelling variants in the printout (key = lowercase variant).
ALIAS = {"arjun": "arjuna", "yudhisthira": "yudhishthira", "sakuntala": "shakuntala",
         "sairandhri(draupadi)": "draupadi", "vishnu (kurma)": "vishnu",
         "martanda varma": "marthanda varma", "bhudevi": "bhoomi devi"}
norm = lambda c: ALIAS.get(c.strip().lower(), c.strip().lower())

table = collections.defaultdict(lambda: {"count": 0, "files": [], "raw_variants": set()})
for k, v in kept.items():
    for c in v["scene_metadata"]["characters"]:
        n = norm(c)
        table[n]["count"] += 1
        table[n]["files"].append(stem(k))
        table[n]["raw_variants"].add(c.strip())

filled = sum(1 for v in kept.values() if v["scene_metadata"]["characters"])
print(f"{len(kept)} training images; {filled} ({100*filled/len(kept):.0f}%) have characters filled")
ge3 = [n for n, t in table.items() if t["count"] >= 3]
print(f"{len(table)} distinct characters; {len(ge3)} appear in >= 3 images")
print("(For the record, the Overview 9.3 bar was >= 60% filled and >= 6 characters in >= 3 images)\n")

print("character | images | coverage | raw spellings seen")
for n, t in sorted(table.items(), key=lambda x: -x[1]["count"]):
    cov = "high" if t["count"] >= 5 else "medium" if t["count"] >= 2 else "low"
    print(f"{n:28s} {t['count']:3d}  {cov:7s} {sorted(t['raw_variants'])}")

out = {n: {"count": t["count"], "files": sorted(t["files"]),
           "raw_variants": sorted(t["raw_variants"])} for n, t in table.items()}
json.dump({"n_images": len(kept), "n_filled": filled, "characters": out},
          open("train/coverage_table.json", "w"), indent=1)
print("\nsaved train/coverage_table.json")
