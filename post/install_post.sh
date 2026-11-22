#!/usr/bin/env bash
# Install GFPGAN on Colab's Python 3.13. Why it fails: basicsr's setup.py reads its version with
# exec(...); locals()['__version__'], which no longer works on Python 3.13 (PEP 667), so the wheel build exits 1.
# Fix: build basicsr from its sdist with a patched setup.py, then install gfpgan without dependencies.
# Usage: bash post/install_post.sh        (prints the working pins at the end)
set -u
T0=$(date +%s)
WORK=$(mktemp -d); cd "$WORK"
# `pip download` of basicsr itself fails (it runs the broken setup.py), so fetch the sdist straight from PyPI
URL=$(curl -s https://pypi.org/pypi/basicsr/json | python -c "import sys,json;d=json.load(sys.stdin);print([u['url'] for u in d['urls'] if u['packagetype']=='sdist'][0])")
curl -sL -o basicsr.tgz "$URL" && tar xzf basicsr.tgz && cd basicsr-*/
python - <<'PY'
import re
s = open("setup.py").read()
s2 = re.sub(r"exec\(compile\(f\.read\(\), version_file, 'exec'\)\)\n    return locals\(\)\['__version__'\]",
            "ns = {}\n        exec(compile(f.read(), version_file, 'exec'), ns)\n    return ns['__version__']", s)
open("setup.py", "w").write(s2)
print("patched setup.py:", s2 != s)
PY
pip install -q --no-build-isolation . 2>&1 | tail -3
# torchvision >= 0.17 removed functional_tensor, which basicsr's degradations.py imports
python - <<'PY'
import glob, site
for sp in site.getsitepackages():
    for f in glob.glob(sp + "/basicsr/data/degradations.py"):
        s = open(f).read()
        if "functional_tensor" in s:
            open(f, "w").write(s.replace("torchvision.transforms.functional_tensor", "torchvision.transforms.functional"))
            print("patched", f)
PY
pip install -q facexlib opencv-python-headless 2>&1 | tail -2
pip install -q --no-deps gfpgan 2>&1 | tail -2
python -c "import basicsr, gfpgan; print('import OK: gfpgan', gfpgan.__version__, 'basicsr', basicsr.__version__)"
pip list 2>/dev/null | grep -iE '^(torch|torchvision|numpy|gfpgan|basicsr|facexlib|opencv-python-headless) '
echo "install took $(( ($(date +%s) - T0) / 60 )) min"
