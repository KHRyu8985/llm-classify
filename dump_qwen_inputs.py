"""Dump original / SoM / mosaic / ICL / VQA figures under results/.../qwen_inputs."""
import argparse, json
from pathlib import Path
from PIL import Image
from som_common import (
    set_dataset, prepare, collect_yolo1, pick_exemplars, label_props,
    pair_images, prompt_ex, PROMPT_ALL, open_rgb, _panel, labels_json, size_note,
    SIZE, stage_qa,
)

N = 3
VQA = (
    ("dedup", "Task 1/4 dedup: keep vs dup (same fruit)"),
    ("filt", "Task 2/4 filt: tomato vs not_tomato (tiny → not_tomato)"),
    ("unripe", "Task 3/4 unripe vs later"),
    ("ripe_semi", "Task 4/4 leftover ripe vs semi_ripe"),
)


def _row(pans):
    out = Image.new("RGB", (sum(p.size[0] for p in pans), max(p.size[1] for p in pans)), (18, 20, 22))
    x = 0
    for p in pans:
        out.paste(p, (x, 0))
        x += p.size[0]
    return out


def save_item(folder, item, prompt, lab=None):
    folder.mkdir(parents=True, exist_ok=True)
    orig = open_rgb(item["path"]).resize((SIZE, SIZE), Image.BILINEAR)
    som, mosaic = pair_images(item)
    orig.save(folder / "input.jpg", quality=90)
    som.save(folder / "som.jpg", quality=90)
    mosaic.save(folder / "mosaic.jpg", quality=90)
    _row([
        _panel(orig, "input"),
        _panel(som, "SoM"),
        _panel(mosaic, "mosaic"),
    ]).save(folder / "strip.jpg", quality=90)
    lab = lab if lab is not None else labels_json(item)
    (folder / "prompt.txt").write_text(prompt + "\n" + size_note(item))
    (folder / "label.json").write_text(json.dumps(lab, indent=2, ensure_ascii=False))
    return lab


def stack(dst, rels, name):
    strips = [Image.open(dst / r / "strip.jpg") for r in rels]
    out = Image.new("RGB", (strips[0].size[0], sum(s.size[1] for s in strips)), (18, 20, 22))
    y = 0
    for s in strips:
        out.paste(s, (0, y))
        y += s.size[1]
    out.save(dst / name, quality=90)


def card(html, title, rel, lab):
    html.append(f"<h2>{title}</h2><img src='{rel}/strip.jpg' width=1400>"
                f"<pre>{json.dumps(lab, ensure_ascii=False)}</pre>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="little")
    args = ap.parse_args()
    set_dataset(args.ds)
    from som_common import OUT
    train, test = prepare(max_eval=N)
    y1 = collect_yolo1(train + test)
    exs = pick_exemplars(train, n=2, box_map=y1)
    query = label_props(test[0], y1.get(test[0]["path"], []))
    dst = OUT / "qwen_inputs"
    html = ["<html><body><h1>VLM input: input | SoM | mosaic</h1>",
            "<p>Eval uses one JSON (ripe/semi_ripe/unripe/not_tomato). "
            "SFT/RL sample one of 4 VQA stages + 1 ICL of that stage.</p>"]
    for i, ex in enumerate(exs):
        rel = f"icl_{i}"
        card(html, f"eval ICL {i} {Path(ex['path']).name}", rel,
             save_item(dst / rel, ex, prompt_ex(i + 1)))
    for i, raw in enumerate(test):
        it = label_props(raw, y1.get(raw["path"], []))
        rel = f"query_{i}"
        card(html, f"eval query {i} YOLO1@0.1 {Path(raw['path']).name}", rel,
             save_item(dst / rel, it, PROMPT_ALL))
    stack(dst, [f"icl_{i}" for i in range(len(exs))] + [f"query_{i}" for i in range(N)],
          "overview.jpg")
    html.append("<h1>SFT / RL VQA stages (same query image)</h1>")
    vqa_rels = []
    for kind, title in VQA:
        q_icl, a_icl = stage_qa(exs[0], kind, example=1)
        q, a = stage_qa(query, kind)
        for sub, item, prompt, lab in (("icl", exs[0], q_icl, a_icl), ("query", query, q, a)):
            rel = f"vqa/{kind}/{sub}"
            save_item(dst / rel, item, prompt, lab)
            card(html, f"{title} — {sub}", rel, lab)
        vqa_rels.append(f"vqa/{kind}/query")
    stack(dst, vqa_rels, "vqa_overview.jpg")
    (dst / "index.html").write_text("".join(html) + "</body></html>")
    print(dst)


if __name__ == "__main__":
    main()
