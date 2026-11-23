"""The Next Ravi Varma: Gradio app (owner D). Scene text + optional pose + style slider -> labelled painting.

Run (Colab, T4, gen profile):   python app/app.py --share [--auth user:password]
- Prompt = M3's default pipeline c1: expand(text, scene, "raw"). No LLM.
- Defaults and slider ranges come from config/m3.json + config/m4.json (app/core.py), never from this file.
- One GPU job at a time (queue). The model is loaded once, at launch.
- Pose: C's module (APP_POSE_MODULE, default pose.stub until S2) -> A's generate_controlled when it is importable;
  before that, the pose is shown but not applied, and the app says so.
- Every image shown or downloaded carries the content label (caption bar + PNG metadata). Evaluation images never
  go through the app, so they are never labelled.
- Uploaded photos are used only to extract a skeleton, deleted by the handler as soon as the skeleton exists, and
  Gradio's own cache is cleared on a timer (delete_cache). docs/APP.md gives the measured N; nothing here claims
  "not saved".
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import core, nsfw, policy  # noqa: E402

POSE_MODULE = os.environ.get("APP_POSE_MODULE", "pose.stub")     # S2: switch to C's real module (one line / env)
POST_MODULE = os.environ.get("APP_POST_MODULE", "post.stub")     # Day 2: switch to D's real post module
CACHE_EVERY_S, CACHE_MAX_AGE_S = 60, 300                         # Gradio delete_cache; N is measured, see APP.md
OUT_DIR = Path(tempfile.gettempdir()) / "nrv_app_outputs"
POSE_MODES = ["none", "preset", "upload photo", "auto from text"]

DEFAULTS = core.defaults()


IMPORT_ERRORS: dict[str, str] = {}      # module name -> why it could not be imported (shown in the server log)


def _import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:       # missing module or missing GPU library: the app degrades, it does not crash
        IMPORT_ERRORS[name] = f"{type(e).__name__}: {e}"
        print(f"[app] could not import {name}: {IMPORT_ERRORS[name]}", file=sys.stderr)   # never silent
        return None


def _presets() -> list[tuple[str, str]]:
    lib = ROOT / "pose" / "library" / "library.json"
    if not lib.exists():
        return []
    items = json.loads(lib.read_text(encoding="utf-8"))
    items = items.get("presets", items) if isinstance(items, dict) else items
    return [(p.get("label", p["name"]), p["name"]) for p in items]


# ---------------------------------------------------------------------------------------------------- engine
class Engine:
    """Loads once. Uses A's generate_controlled when available, M3's generate otherwise."""

    def __init__(self, warm: bool = True):
        self.gen = importlib.import_module("pipeline.generate_api")
        self.control = _import("pipeline.control_api")
        self.pose = _import(POSE_MODULE)
        self.post = _import(POST_MODULE)
        self.nsfw = nsfw.Classifier()                            # loads on first use, on CPU
        if self.control is None:
            print("[app] POSE CONTROL IS OFF: pipeline.control_api did not import (reason above). Poses will be "
                  "shown but not applied.", file=sys.stderr)
        from kb.parser import parse_scene
        from expander.expand import expand
        self.parse_scene, self.expand = parse_scene, expand
        if warm:
            self.gen._ensure(DEFAULTS["model"], None)

    def control_image(self, mode, preset, photo_path, scene, text):
        """-> (ControlImage | None, warnings). Never raises: falls back to no pose with a message."""
        if mode == "none":
            return None, []
        if self.pose is None:
            return None, ["Pose input is not available in this build, so no pose was used."]
        try:
            if mode == "preset":
                if not preset:
                    return None, ["Pick a preset pose first; no pose was used."]
                ctrl = self.pose.make_pose(preset)
            elif mode == "upload photo":
                if not photo_path:
                    return None, ["Upload a photo first; no pose was used."]
                from PIL import Image
                with Image.open(photo_path) as im:
                    ctrl = self.pose.extract_pose(im.convert("RGB"))
            else:
                ctrl = self.pose.pose_from_text(scene if scene else text)
        except Exception as e:                                   # noqa: BLE001
            return None, [f"Pose extraction failed ({type(e).__name__}), so no pose was used."]
        finally:
            if photo_path:
                _delete(photo_path)                              # the photo is not needed once we have a skeleton
        warnings = list(ctrl.get("warnings", [])) if isinstance(ctrl, dict) else []
        if isinstance(ctrl, dict) and ctrl.get("n_people", 1) == 0:
            return None, warnings + ["No person was found, so no pose was used."]
        return ctrl, warnings

    def generate(self, positive, negative, seed, style, ctrl, pose_strength):
        if ctrl is not None and self.control is not None:
            gs, ge = DEFAULTS["pose_window"]                      # window of the active (tuned or baseline) profile
            return self.control.generate_controlled(positive, negative, seed, ctrl, model=DEFAULTS["model"],
                                                    lora_scale=style, control_scale=pose_strength,
                                                    guidance_start=gs, guidance_end=ge), True
        return self.gen.generate(positive, negative, seed, model=DEFAULTS["model"], lora_scale=style), False

    def nsfw_score(self, image):
        clf = getattr(self, "nsfw", None)
        return clf.score(image) if clf else None

    def postprocess(self, image, positive, negative, seed, hires, restore, style=None):
        done = []
        if self.post is None:
            return image, done
        if hires:
            image = self.post.hires(image, positive, negative, seed, lora_scale=style); done.append("hires")
        if restore:
            image = self.post.restore_faces(image); done.append("restore_faces")
        return image, done


def _delete(path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------------- handler
def make_handler(engine: Engine):
    def run(text, pose_mode, preset, photo_path, style, pose_strength, seed, post_hires, post_restore):
        """-> (labelled image, skeleton, details markdown, settings json, download path, status markdown)"""
        empty = (None, None, "", "", None)
        try:
            chk = core.check_text(text)
            if not chk["ok"]:
                if photo_path:
                    _delete(photo_path)
                return (*empty, chk["message"])
            pol = policy.check(text)
            if not pol["allowed"]:
                if photo_path:
                    _delete(photo_path)
                return (*empty, pol["message"])
            seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
            scene = engine.parse_scene(text)
            prompt = engine.expand(text, scene, "raw")
            pol = policy.check(prompt["positive"])
            if not pol["allowed"]:
                return (*empty, pol["message"])
            ctrl, warnings = engine.control_image(pose_mode, preset, photo_path, scene, text)
            image, pose_used = engine.generate(prompt["positive"], prompt["negative"], seed, style, ctrl,
                                               pose_strength)
            if ctrl is not None and not pose_used:
                warnings.append("Pose control is not available on this server right now, so the pose is shown "
                                "but was not applied.")
            image, post_done = engine.postprocess(image, prompt["positive"], prompt["negative"], seed,
                                                  post_hires, post_restore, style)
            settings = core.settings_record(text, prompt, seed, style, pose_mode,
                                            pose_strength if pose_used else None,
                                            (ctrl or {}).get("source") if ctrl else None, post_done,
                                            DEFAULTS["model"], DEFAULTS["pose_window"] if pose_used else None)
            labelled = core.add_label(image)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            clear_path = OUT_DIR / f"nrv_seed{seed}_clear.png"
            clear_path.unlink(missing_ok=True)                   # never leave a stale image from an earlier run
            score_fn = getattr(engine, "nsfw_score", None)
            score = score_fn(image) if score_fn else None
            flagged = nsfw.is_flagged(score)
            if score is not None:
                settings["nsfw_flagged"] = flagged
            if flagged:                                          # soft filter: blur, keep the clear copy for "Show"
                core.save_png(labelled, clear_path, settings)
                labelled = core.add_label(nsfw.blur(image))
            path = core.save_png(labelled, OUT_DIR / f"nrv_seed{seed}.png", settings)
            chars = [c.get("canonical") for c in (scene or {}).get("characters", [])]
            details = (f"**Prompt:** {prompt['positive']}\n\n**Negative:** {prompt['negative']}\n\n"
                       f"**Characters found:** {', '.join(chars) or 'none'}\n\n**Seed:** {seed}")
            if flagged:
                warnings.append("The safety filter flagged this image, so it is blurred. Press \"Show image\" to view "
                                "it. The filter is a backstop and is often wrong on paintings (nursing, bathing, "
                                "drapery).")
            notes = [chk["warning"]] + warnings
            status = "Done." + ("".join(f"\n\n- {n}" for n in notes if n))
            skeleton = ctrl.get("image") if isinstance(ctrl, dict) else None
            return labelled, skeleton, details, json.dumps(settings, indent=1, ensure_ascii=False), str(path), status
        except Exception as e:                                   # noqa: BLE001
            if photo_path:
                _delete(photo_path)
            if type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower():
                _free_gpu()
                return (*empty, "The GPU ran out of memory. Please try again, ideally without the post-processing "
                                "options.")
            traceback.print_exc()
            return (*empty, f"Something went wrong ({type(e).__name__}). Please try again; if it keeps happening, "
                            "try a shorter description or no pose.")
    return run


def _free_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                            # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------------------------- UI
HELP_STYLE = "Higher = closer to Ravi Varma's painted look. Only tested values are offered."
HELP_POSE = ("Higher = follows the pose more closely, but may look stiffer. Only tested values are offered.")
ABOUT = ("**About.** Research and coursework project. Images are AI-generated *in the style of* Raja Ravi Varma and "
         "are never authentic works; don't present them as his. Deities and epic figures are depicted respectfully: "
         "sexualised, demeaning, hateful, gory content and real living people are refused (docs/CONTENT_POLICY.md).")
UPLOAD_NOTE = ("Your photo is used only to extract a stick-figure pose, and the app deletes it as soon as the pose "
               "exists; Gradio's own upload cache is cleared within about {n} minutes.")


def reveal(settings_json: str):
    """The "Show image" button: returns the unblurred copy that make_handler saved for a flagged image."""
    try:
        seed = int(json.loads(settings_json)["seed"])
        from PIL import Image
        with Image.open(OUT_DIR / f"nrv_seed{seed}_clear.png") as im:
            return im.convert("RGB")
    except Exception:                                            # noqa: BLE001
        return None


def _slider(gr, rng, value, label, info):
    """Only tested values: the range comes from config. A one-value range is shown fixed (not editable)."""
    lo, hi = float(rng[0]), float(rng[-1])
    if hi <= lo:
        return gr.Slider(lo, lo + 0.05, value=lo, step=0.05, label=label, info=info + " (fixed until tested)",
                         interactive=False)
    return gr.Slider(lo, hi, value=value, step=0.05, label=label, info=info)


def build(engine: Engine | None):
    import gradio as gr
    d = DEFAULTS
    presets = _presets()
    run = make_handler(engine) if engine else (lambda *a: (None, None, "", "", None, "Model not loaded."))
    with gr.Blocks(title="The Next Ravi Varma", delete_cache=(CACHE_EVERY_S, CACHE_MAX_AGE_S)) as demo:
        gr.Markdown("# The Next Ravi Varma\nDescribe a scene from the epics; optionally give a pose.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Scene", lines=3,
                                  placeholder="Sita sitting alone under a tree in the Ashoka grove, sorrowful")
                pose_mode = gr.Radio(POSE_MODES, value="none", label="Pose")
                preset = gr.Dropdown(presets, label="Preset pose", visible=False)
                photo = gr.Image(type="filepath", sources=["upload"], label="Pose photo", visible=False)
                photo_note = gr.Markdown(UPLOAD_NOTE.format(n=round(CACHE_MAX_AGE_S / 60 + CACHE_EVERY_S / 60)),
                                         visible=False)
                style = _slider(gr, d["style_range"], d["style"], "Style strength", HELP_STYLE)
                pose_strength = _slider(gr, d["pose_range"], d["pose"], "Pose strength", HELP_POSE)
                with gr.Row():
                    seed = gr.Number(value=1, precision=0, label="Seed")
                    rnd = gr.Button("Randomise seed")
                post_hires = gr.Checkbox(value=d["post_default_on"], label="Higher resolution (slower)")
                post_restore = gr.Checkbox(value=False, label="Face restoration (may look less painted)")
                go = gr.Button("Generate", variant="primary")
                gr.Markdown(core.time_note())
            with gr.Column():
                out = gr.Image(type="pil", label="Painting", format="png")
                status = gr.Markdown()
                show = gr.Button("Show image", visible=False)
                download = gr.File(label="Download (PNG with label and settings in the metadata)")
                with gr.Accordion("Show details", open=False):
                    skeleton = gr.Image(type="pil", label="Pose used")
                    details = gr.Markdown()
                    settings = gr.Code(label="Copy settings (reproduce from the CLI)", language="json")
        gr.Markdown(ABOUT)

        def _modes(m):
            return (gr.update(visible=m == "preset"), gr.update(visible=m == "upload photo"),
                    gr.update(visible=m == "upload photo"))
        pose_mode.change(_modes, pose_mode, [preset, photo, photo_note])
        rnd.click(lambda: random.randint(1, 2**31 - 1), None, seed)

        def on_go(*a):
            r = run(*a)
            try:
                flagged = bool(json.loads(r[3]).get("nsfw_flagged"))
            except Exception:                                    # noqa: BLE001
                flagged = False
            return (*r, gr.update(visible=flagged))
        go.click(on_go, [text, pose_mode, preset, photo, style, pose_strength, seed, post_hires, post_restore],
                 [out, skeleton, details, settings, download, status, show])
        show.click(reveal, settings, out)
    demo.queue(max_size=4, default_concurrency_limit=1)
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public Gradio link (protect it with --auth)")
    ap.add_argument("--auth", default=os.environ.get("APP_AUTH"), help="user:password (or env APP_AUTH)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no_model", action="store_true", help="build the UI without loading the model (UI check)")
    a = ap.parse_args(argv)
    engine = None if a.no_model else Engine(warm=True)
    demo = build(engine)
    auth = tuple(a.auth.split(":", 1)) if a.auth else None
    demo.launch(share=a.share, auth=auth, server_port=a.port, show_error=False)


if __name__ == "__main__":
    main()
