"""Copy visual refs for tutorial/README.md (photos only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shutil
from PIL import Image, ImageDraw
from som_common import (
    set_dataset, prepare, collect_yolo1, label_props, pair_images,
    open_rgb, SIZE, COLORS, SMALL_FONT, _scale_boxes, _badge, iou,
)


SRC = Path("results/som_little/qwen_inputs")
DST = Path("tutorial/img")
FILES = {
    "query_input.jpg": "query_0/input.jpg",
    "query_som.jpg": "query_0/som.jpg",
    "query_mosaic.jpg": "query_0/mosaic.jpg",
    "vqa_filt.jpg": "vqa/filt/query/strip.jpg",
    "vqa_unripe.jpg": "vqa/unripe/query/strip.jpg",
    "vqa_ripe_semi.jpg": "vqa/ripe_semi/icl/strip.jpg",
}


def ids_only(item):
    orig = open_rgb(item["path"]).resize((SIZE, SIZE), Image.BILINEAR)
    boxes = _scale_boxes(item["boxes"], item["wh"], SIZE)
    d = ImageDraw.Draw(orig)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        c = COLORS[i % len(COLORS)]
        cx = max(2, min((x1 + x2) // 2, SIZE - 28))
        cy = max(2, min((y1 + y2) // 2, SIZE - 20))
        _badge(d, (cx, cy), str(i + 1), c, SMALL_FONT)
    return orig


def dump_dedup(train, y1):
    for it in train:
        lab = label_props(it, y1.get(it["path"], []))
        dups = [i for i, l in enumerate(lab["labels"]) if l == "dup"]
        if len(dups) < 2:
            continue
        som, mosaic = pair_images(lab)
        som.save(DST / "vqa_dedup.jpg", quality=90)
        mosaic.save(DST / "vqa_dedup_mosaic.jpg", quality=90)
        # zoom around first dup vs its overlap
        boxes = lab["boxes"]
        i = dups[0]
        js = [j for j in range(len(boxes)) if j != i and iou(boxes[i], boxes[j]) > 0.1]
        if not js:
            continue
        ids = [i] + js[:2]
        xs = [boxes[k][t] for k in ids for t in (0, 2)]
        ys = [boxes[k][t] for k in ids for t in (1, 3)]
        pad = 40
        crop = [max(0, min(xs) - pad), max(0, min(ys) - pad),
                min(lab["wh"][0], max(xs) + pad), min(lab["wh"][1], max(ys) + pad)]
        orig = open_rgb(lab["path"])
        z = orig.crop(crop).resize((SIZE, SIZE), Image.BILINEAR)
        sx = SIZE / (crop[2] - crop[0])
        sy = SIZE / (crop[3] - crop[1])
        d = ImageDraw.Draw(z)
        gold = {i + 1: ("dup" if lab["labels"][i] == "dup" else "keep") for i in range(len(boxes))}
        for k in ids:
            x1, y1, x2, y2 = boxes[k]
            x1, y1 = int((x1 - crop[0]) * sx), int((y1 - crop[1]) * sy)
            x2, y2 = int((x2 - crop[0]) * sx), int((y2 - crop[1]) * sy)
            c = COLORS[k % len(COLORS)]
            d.rectangle([x1, y1, x2, y2], outline=c, width=4)
            _badge(d, (max(2, x1), max(2, y1)), f"{k+1}:{gold[k+1]}", c, SMALL_FONT)
        z.save(DST / "vqa_dedup_zoom.jpg", quality=90)
        print("dedup", Path(lab["path"]).name, "n_dup", len(dups),
              "ids", [k + 1 for k in ids])
        return
    raise SystemExit("no dup example")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for name in DST.glob("*"):
        name.unlink()
    for dst, src in FILES.items():
        shutil.copy2(SRC / src, DST / dst)
    set_dataset("little")
    train, test = prepare(max_eval=1)
    y1 = collect_yolo1(train + test)
    dump_dedup(train, y1)
    ids_only(label_props(test[0], y1[test[0]["path"]])).save(DST / "query_ids_only.jpg", quality=90)
    print("ok", DST)


if __name__ == "__main__":
    main()
