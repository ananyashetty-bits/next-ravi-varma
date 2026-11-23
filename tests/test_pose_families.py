import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_labels_match_pairs_and_allowed_families():
    fam = json.load(open(ROOT / "eval" / "pose_families_dev.json"))
    pairs = {p["id"]: p for p in json.load(open(ROOT / "eval" / "test_pairs_dev.json"))}
    assert set(fam["labels"]) == set(pairs), "label file and test_pairs_dev.json must list the same pairs"
    for sid, labs in fam["labels"].items():
        assert len(labs) == pairs[sid]["n_people_expected"], sid
        assert all(l in fam["families"] for l in labs), sid
    assert set(fam["uncertain"]) <= set(fam["labels"])
