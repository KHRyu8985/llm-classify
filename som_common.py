"""Shared SoM helpers: marks, mosaic, labels, eval."""
import json, math, random, re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
SIZE = 768
N_DISTRACTOR = 2
CLASSES = ("ripe", "semi_ripe", "unripe", "other")
CAT_MAP = {
    1: "ripe", 2: "semi_ripe", 3: "unripe",
    4: "ripe", 5: "semi_ripe", 6: "unripe",
}
ALIAS = {
    "ripe": "ripe", "fully_ripened": "ripe", "fully-ripened": "ripe",
    "red": "ripe", "fully ripe": "ripe",
    "semi_ripe": "semi_ripe", "semiripe": "semi_ripe", "half_ripened": "semi_ripe",
    "half-ripened": "semi_ripe", "semi": "semi_ripe", "half": "semi_ripe",
    "unripe": "unripe", "green": "unripe",
    "other": "other", "not_tomato": "other", "none": "other",
    "no_tomato": "other", "no-tomato": "other", "not tomato": "other",
    "dup": "other", "duplicate": "other", "same": "other",
    "small": "other", "tiny": "other", "too_small": "other",
}
COLORS = [
    (255, 56, 56), (56, 200, 80), (56, 120, 255), (255, 200, 40),
    (200, 80, 255), (40, 220, 220), (255, 120, 40), (180, 255, 60),
]
FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
)
SMALL_FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
)
QUAL_FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
)
DATASETS = {
    "kaggle": dict(
        kind="coco", root=Path("./laboro-tomato"),
        img={"train": "train/images", "test": "val/images"},
        ann={"train": "train.json", "test": "test.json"},
        out=Path("./results/som_kaggle"),
        yolo1="weights/kaggle/yolo11_1cls.pt",
        yolo3="weights/kaggle/yolo11.pt",
        yolo_img=Path("data/yolo/laboro_kaggle_3cls/test/images"),
    ),
    "little": dict(
        kind="coco", root=Path("./data/raw/laboro_tomato_little/laboro_little"),
        img={"train": "train", "test": "test"},
        ann={"train": "train.json", "test": "test.json"},
        out=Path("./results/som_little"),
        yolo1="weights/little/yolo11_1cls.pt",
        yolo3="weights/little/yolo11.pt",
        yolo_img=Path("data/yolo/laboro_little_yolo_3cls/test/images"),
    ),
    "laboro_tomatod": dict(
        kind="yolo", root=Path("./data/yolo/laboro_tomatod_yolo_3cls"),
        out=Path("./results/som_tomatod"),
        yolo1="weights/laboro_tomatod/yolo11_1cls.pt",
        yolo3="weights/laboro_tomatod/yolo11.pt",
        yolo_img=Path("data/yolo/laboro_tomatod_yolo_3cls/test/images"),
    ),
}
ROOT = DATASETS["little"]["root"]
OUT = DATASETS["little"]["out"]
OUT.mkdir(parents=True, exist_ok=True)
YOLO_1CLS_W = DATASETS["little"]["yolo1"]
YOLO_3CLS_W = DATASETS["little"]["yolo3"]
YOLO_IMG = DATASETS["little"]["yolo_img"]
KIND, IMG, ANN = DATASETS["little"]["kind"], DATASETS["little"].get("img"), DATASETS["little"].get("ann")
DET_CONF = 0.05
PROP_CONF = 0.1
MAX_PROP = 20
TOMATO = ("ripe", "semi_ripe", "unripe")
SMALL_REL = 0.00084
SMALL_SIDE = 0.03
NEED3 = set(TOMATO)
YOLO3_NAMES = {0: "ripe", 1: "semi_ripe", 2: "unripe"}
CLS_COL = {
    "ripe": (220, 40, 40), "semi_ripe": (255, 160, 0), "unripe": (30, 170, 50),
    "yolo1": (0, 190, 255),     "not_tomato": (170, 170, 170),
}


def _make_prompt(n, rel5, rel50, rel95, side5, side50, px5):
    return f"""You are a senior greenhouse tomato-ripeness annotator.

For every mark, use BOTH images. Switch between them:
- Image 1 (SoM): full frame with numbered boxes. Use this for cluster context — a tomato behind another tomato still counts.
- Image 2 (mosaic): zoomed crop of that box. Tile order is shuffled; empty cells are black. Match by the number badge, never by grid position.
Mosaic = skin color. SoM = is there a fruit in that box, including fruit hidden behind another.

Ripeness (fruit skin, ignore calyx/leaf):
- ripe: mostly red
- semi_ripe: orange, yellow, or clearly mixed red and green
- unripe: green or whitish
- not_tomato: leaf, stem, flower, twine, or empty background — no fruit at all

Keep the tomato if any fruit is visible, even behind another tomato, loose box, or occluded. If unsure, pick a ripeness class, not not_tomato.
not_tomato only when you are sure there is no tomato in the box.
Labeled GT size (train n={n}): box-area/image p5={100 * rel5:.3f}% p50={100 * rel50:.2f}% p95={100 * rel95:.1f}%; min-side/min(image) p5={100 * side5:.1f}% p50={100 * side50:.1f}% (~{px5:.0f}px at p5). Annotators deliberately skipped fruit below ~p5. Mosaic zoom makes tiny crops look large — use SoM size and the area list. If a mark is below that p5 range, not_tomato even if it looks like fruit.
If several numbered boxes sit on the SAME fruit in the SoM, keep one tighter box and treat extras as duplicates — that is not two tomatoes.

Do not list marks one by one. Classify this image's numbers."""


PROMPT = _make_prompt(7781, 0.00084, 0.00807, 0.04908, 0.0304, 0.0967, 94)
ICL_HINT = (
    "Previous turns are labeled in-context examples. Follow that format. "
    "This is a NEW image; do not copy those labels."
)
PROMPT_ALL = (
    PROMPT + "\n\n" + ICL_HINT
    + "\nClassify every mark. JSON only:\n"
    + '{"1":"ripe","2":"not_tomato"}'
)


def set_dataset(name="little"):
    global ROOT, OUT, YOLO_1CLS_W, YOLO_3CLS_W, YOLO_IMG, KIND, IMG, ANN
    cfg = DATASETS[name]
    KIND, ROOT, OUT = cfg["kind"], cfg["root"], cfg["out"]
    YOLO_1CLS_W, YOLO_3CLS_W, YOLO_IMG = cfg["yolo1"], cfg["yolo3"], cfg["yolo_img"]
    IMG, ANN = cfg.get("img"), cfg.get("ann")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"dataset={name} kind={KIND} root={ROOT} out={OUT}")


def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def apply_size_stats(items):
    global SMALL_REL, SMALL_SIDE, PROMPT, PROMPT_ALL
    rel, side, px = [], [], []
    for it in items:
        iw, ih = max(1, it["wh"][0]), max(1, it["wh"][1])
        for b in it["boxes"]:
            w, h = max(1, b[2] - b[0]), max(1, b[3] - b[1])
            rel.append(w * h / (iw * ih))
            side.append(min(w, h) / min(iw, ih))
            px.append(min(w, h))
    SMALL_REL, SMALL_SIDE = _pct(rel, 5), _pct(side, 5)
    PROMPT = _make_prompt(
        len(rel), SMALL_REL, _pct(rel, 50), _pct(rel, 95),
        SMALL_SIDE, _pct(side, 50), _pct(px, 5),
    )
    PROMPT_ALL = (
        PROMPT + "\n\n" + ICL_HINT
        + "\nClassify every mark. JSON only:\n"
        + '{"1":"ripe","2":"not_tomato"}'
    )
    (OUT / "prompt.txt").write_text(PROMPT_ALL)
    print(f"GT size n={len(rel)} p5_rel={SMALL_REL:.5f} p5_side={SMALL_SIDE:.4f}")


def cat_lab(cid):
    return CAT_MAP.get(cid) or YOLO3_NAMES.get(int(cid), "unripe")


def prompt_ex(k=1):
    return (
        PROMPT
        + f"\n\nThis is worked EXAMPLE {k} with gold labels. JSON only:\n"
        + '{"1":"ripe","2":"not_tomato"}'
    )


def size_note(item):
    iw, ih = max(1, item["wh"][0]), max(1, item["wh"][1])
    xs = []
    for i, b in enumerate(item["boxes"]):
        w, h = max(1, b[2] - b[0]), max(1, b[3] - b[1])
        xs.append(f"{i + 1}:{100 * w * h / (iw * ih):.2f}%")
    return "This image mark area/image: " + " ".join(xs)


def pub(lab):
    return {"other": "not_tomato", "small": "not_tomato", "dup": "dup"}.get(lab, lab)


def open_rgb(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) or img
    return img.convert("RGB")


def _xyxy(bbox):
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


def too_small(box, wh):
    w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    iw, ih = max(1, wh[0]), max(1, wh[1])
    return (w * h) / (iw * ih) < SMALL_REL or min(w, h) / min(iw, ih) < SMALL_SIDE


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def load_split(split):
    if KIND == "yolo":
        return _load_yolo(split)
    key = "train" if split == "train" else "test"
    ann = json.loads((ROOT / "annotations" / ANN[key]).read_text())
    img_dir = ROOT / IMG[key]
    info = {i["id"]: i for i in ann["images"]}
    by = defaultdict(list)
    for a in ann["annotations"]:
        by[a["image_id"]].append(a)
    items = []
    for iid, anns in by.items():
        p = img_dir / info[iid]["file_name"]
        if not p.exists():
            continue
        items.append({
            "path": str(p),
            "boxes": [_xyxy(a["bbox"]) for a in anns],
            "labels": [cat_lab(a["category_id"]) for a in anns],
            "wh": (info[iid]["width"], info[iid]["height"]),
        })
    return items


def _load_yolo(split):
    img_dir = ROOT / split / "images"
    if split == "test" and not img_dir.exists():
        img_dir = ROOT / "val" / "images"
    lab_dir = img_dir.parent / "labels"
    items = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        w, h = Image.open(p).size
        boxes, labels = [], []
        lp = lab_dir / (p.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                c, xc, yc, bw, bh = line.split()[:5]
                c, xc, yc, bw, bh = int(float(c)), *map(float, (xc, yc, bw, bh))
                x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
                boxes.append([int(x1), int(y1), int(x1 + bw * w), int(y1 + bh * h)])
                labels.append(cat_lab(c))
        items.append({"path": str(p), "boxes": boxes, "labels": labels, "wh": (w, h)})
    return items


def prepare(max_eval=32, max_train=0):
    train = load_split("train")
    if max_train:
        train = train[:max_train]
    apply_size_stats(train)
    return train, load_split("test")[:max_eval]


def add_distractors(item, k=N_DISTRACTOR, rng=None):
    rng = rng or random.Random(0)
    w, h = item["wh"]
    boxes = list(item["boxes"])
    labels = list(item["labels"])
    for _ in range(80):
        if sum(l == "other" for l in labels) >= k:
            break
        bw, bh = rng.randint(80, 240), rng.randint(80, 240)
        x1 = rng.randint(0, max(1, w - bw))
        y1 = rng.randint(0, max(1, h - bh))
        box = [x1, y1, x1 + bw, y1 + bh]
        if all(iou(box, b) < 0.05 for b in boxes):
            boxes.append(box)
            labels.append("other")
    item = dict(item)
    item["boxes"], item["labels"] = boxes, labels
    return shuffle_marks(item, rng)


def nms_merge(boxes, thr=0.7):
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep = []
    for b in boxes:
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def detect_xyxy(model, item, conf=DET_CONF, device=0, max_n=MAX_PROP):
    import numpy as np
    im = open_rgb(item["path"])
    arr = np.ascontiguousarray(np.array(im)[:, :, ::-1])
    r = model.predict(
        arr, imgsz=640, verbose=False, conf=conf, iou=0.5, agnostic_nms=True, device=device,
    )[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xy = r.boxes.xyxy.cpu().tolist()
    cf = r.boxes.conf.cpu().tolist()
    sx, sy = item["wh"][0] / max(1, im.size[0]), item["wh"][1] / max(1, im.size[1])
    scored = [([x1 * sx, y1 * sy, x2 * sx, y2 * sy], c) for (x1, y1, x2, y2), c in zip(xy, cf)]
    scored.sort(key=lambda x: -x[1])
    return [[int(v) for v in b] for b, _ in scored[:max_n]]


def detect_cls(model, item, names, conf=0.25, device="cpu", max_n=80):
    import numpy as np
    im = open_rgb(item["path"])
    arr = np.ascontiguousarray(np.array(im)[:, :, ::-1])
    r = model.predict(arr, imgsz=640, verbose=False, conf=conf, iou=0.5, device=device)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    sx, sy = item["wh"][0] / max(1, im.size[0]), item["wh"][1] / max(1, im.size[1])
    out = []
    for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist()):
        out.append(([int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)], names[int(c)]))
        if len(out) >= max_n:
            break
    return out


def collect_yolo1(items, device=0, conf=PROP_CONF):
    from tqdm import tqdm
    from ultralytics import YOLO
    cache_p = OUT / "yolo1_01.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    need = [it for it in items if it["path"] not in cache]
    if not need:
        return cache
    m = YOLO(YOLO_1CLS_W)
    for it in tqdm(need, desc="yolo1"):
        cache[it["path"]] = detect_xyxy(m, it, conf=conf, device=device)
    cache_p.write_text(json.dumps(cache))
    del m
    return cache


def label_props(item, boxes, thr=0.5):
    labels, used, kept = [], set(), []
    for pb in boxes:
        pb = [int(x) for x in pb]
        if too_small(pb, item["wh"]):
            labels.append("small")
            continue
        best_free, biou = -1, thr
        best_any, a_iou = -1, thr
        for i, gb in enumerate(item["boxes"]):
            v = iou(pb, gb)
            if v > a_iou:
                best_any, a_iou = i, v
            if i not in used and v > biou:
                best_free, biou = i, v
        if best_free >= 0:
            used.add(best_free)
            labels.append(item["labels"][best_free])
            kept.append(pb)
        elif best_any >= 0 or any(iou(pb, kb) >= 0.5 for kb in kept):
            labels.append("dup")
        else:
            labels.append("other")
    out = dict(item)
    out["boxes"], out["labels"] = [list(map(int, b)) for b in boxes], labels
    return shuffle_marks(out, random.Random(_item_seed(item, 3)))


def collect_yolo_boxes(items, device=0, conf=DET_CONF):
    from tqdm import tqdm
    from ultralytics import YOLO
    cache_p = OUT / "yolo_props.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    need = [it for it in items if it["path"] not in cache]
    if not need:
        return cache
    ms = [YOLO(YOLO_3CLS_W), YOLO(YOLO_1CLS_W)]
    for it in tqdm(need, desc="yolo-props"):
        boxes = []
        for m in ms:
            boxes.extend(detect_xyxy(m, it, conf=conf, device=device))
        cache[it["path"]] = nms_merge(boxes)
    cache_p.write_text(json.dumps(cache))
    del ms
    return cache


def _near_box(gb, w, h, rng):
    x1, y1, x2, y2 = [int(v) for v in gb]
    bw, bh = max(20, x2 - x1), max(20, y2 - y1)
    ox = rng.choice((-1, 1)) * int(bw * rng.uniform(0.7, 1.3))
    oy = rng.randint(-max(1, bh // 2), max(1, bh // 2))
    nx1 = min(max(0, x1 + ox), max(0, w - bw))
    ny1 = min(max(0, y1 + oy), max(0, h - bh))
    return [nx1, ny1, nx1 + bw, ny1 + bh]


def enrich_item(item, pred_boxes, rng=None, n_rand=2, n_fp=4, max_n=20):
    """GT + YOLO-matched boxes + unmatched YOLO (other) + nearby random."""
    rng = rng or random.Random(0)
    w, h = item["wh"]
    used, boxes, labels, fps = set(), [], [], []
    for pb in pred_boxes:
        best, biou = -1, 0.5
        for i, gb in enumerate(item["boxes"]):
            if i in used:
                continue
            v = iou(pb, gb)
            if v > biou:
                best, biou = i, v
        if best >= 0:
            used.add(best)
            boxes.append([int(x) for x in pb])
            labels.append(item["labels"][best])
        else:
            fps.append(pb)
    for i, gb in enumerate(item["boxes"]):
        if i not in used:
            boxes.append(gb)
            labels.append(item["labels"][i])
    for pb in fps[:n_fp]:
        boxes.append([int(x) for x in pb])
        labels.append("other")
    n_other = n_fp + n_rand
    for _ in range(80):
        if sum(l == "other" for l in labels) >= n_other:
            break
        gb = item["boxes"][rng.randrange(len(item["boxes"]))] if item["boxes"] else [0, 0, 80, 80]
        box = _near_box(gb, w, h, rng)
        if all(iou(box, b) < 0.05 for b in boxes):
            boxes.append(box)
            labels.append("other")
    while len(boxes) > max_n:
        i = next((j for j in range(len(labels) - 1, -1, -1) if labels[j] == "other"), None)
        if i is None:
            break
        boxes.pop(i)
        labels.pop(i)
    out = dict(item)
    out["boxes"], out["labels"] = boxes, labels
    return shuffle_marks(out, rng)


def shuffle_marks(item, rng):
    idx = list(range(len(item["boxes"])))
    rng.shuffle(idx)
    out = dict(item)
    out["boxes"] = [item["boxes"][i] for i in idx]
    out["labels"] = [item["labels"][i] for i in idx]
    return out


def _scale_boxes(boxes, wh, size=SIZE):
    sx, sy = size / wh[0], size / wh[1]
    return [[int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)] for x1, y1, x2, y2 in boxes]


def _badge(draw, xy, text, color, font):
    x, y = xy
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    pad = 4
    draw.rounded_rectangle(
        [x, y, x + tw + pad * 2, y + th + pad * 2], radius=6, fill=color
    )
    draw.text((x + pad, y + pad - 1), text, fill=(0, 0, 0), font=font)


def draw_som(item, size=SIZE, orig=None):
    img = (orig or open_rgb(item["path"])).resize((size, size), Image.BILINEAR)
    boxes = _scale_boxes(item["boxes"], item["wh"], size)
    draw = ImageDraw.Draw(img)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        c = COLORS[i % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        cx = max(2, min(x1 + 2, size - 36))
        cy = max(2, min(y1 + 2, size - 24))
        _badge(draw, (cx, cy), str(i + 1), c, SMALL_FONT)
    return img


def _item_seed(item, salt=0):
    return (sum(map(ord, str(item.get("path", "")))) * 131 + len(item["boxes"]) * 17 + salt) & 0xFFFFFFFF


def make_mosaic(item, size=SIZE, orig=None):
    orig = orig or open_rgb(item["path"])
    n = max(1, len(item["boxes"]))
    grid = math.ceil(math.sqrt(n))
    cell = size // grid
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    order = list(range(n))
    random.Random(_item_seed(item, 7)).shuffle(order)
    for pos, i in enumerate(order):
        r, c = divmod(pos, grid)
        crop = orig.crop(item["boxes"][i]).resize((cell - 6, cell - 6), Image.BILINEAR)
        d = ImageDraw.Draw(crop)
        col = COLORS[i % len(COLORS)]
        d.rectangle([0, 0, crop.size[0] - 1, crop.size[1] - 1], outline=col, width=3)
        _badge(d, (4, 4), str(i + 1), col, FONT)
        canvas.paste(crop, (c * cell + 3, r * cell + 3))
    return canvas


def pair_images(item):
    orig = open_rgb(item["path"])
    return draw_som(item, orig=orig), make_mosaic(item, orig=orig)


def extract_json(s):
    s = s.replace("```json", "").replace("```", "").strip()
    start = s.find("{")
    if start < 0:
        raise ValueError(s)
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError(s)


def norm_label(v):
    if not isinstance(v, str):
        v = str(v)
    v = v.strip().lower().replace(" ", "_")
    return ALIAS.get(v, v if v in CLASSES else None)


def parse_mark_map(text, n):
    obj = extract_json(text)
    if isinstance(obj.get("labels"), dict):
        obj = obj["labels"]
    out = {}
    for k, v in obj.items():
        m = re.search(r"\d+", str(k))
        if not m:
            continue
        idx = int(m.group())
        lab = norm_label(v)
        if lab:
            out[idx] = lab
    return {i: out.get(i, "parse_error") for i in range(1, n + 1)}


def parse_single(text):
    obj = extract_json(text)
    return norm_label(obj.get("tomato") or obj.get("label") or next(iter(obj.values())))


def load_vlm(model_id=MODEL_ID, device="cuda", merge_lora=None):
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": device},
    )
    proc = AutoProcessor.from_pretrained(model_id)
    if merge_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, merge_lora)
        model = model.merge_and_unload()
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    return model, proc


def _apply(proc, msgs, gen=True):
    kw = dict(tokenize=False, add_generation_prompt=gen)
    try:
        return proc.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        return proc.apply_chat_template(msgs, **kw)


def generate(model, proc, messages, max_new_tokens=192, do_sample=False, temperature=1.0):
    import torch
    from qwen_vl_utils import process_vision_info
    text = _apply(proc, messages, True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, return_tensors="pt")
    inputs = inputs.to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, use_cache=True)
    if do_sample:
        kw.update(temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return proc.decode(gen, skip_special_tokens=True).strip(), inputs, out


def generate_batch(model, proc, batch, max_new_tokens=192, do_sample=False, temperature=1.0):
    import torch
    from qwen_vl_utils import process_vision_info
    proc.tokenizer.padding_side = "left"
    texts, imgs = [], []
    for msgs in batch:
        texts.append(_apply(proc, msgs, True))
        im, _ = process_vision_info(msgs)
        imgs.extend(im or [])
    inputs = proc(text=texts, images=imgs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, use_cache=True)
    if do_sample:
        kw.update(temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    plen = inputs["input_ids"].shape[1]
    return [proc.decode(row[plen:], skip_special_tokens=True).strip() for row in out]


def encode_sft(proc, messages, device):
    from qwen_vl_utils import process_vision_info
    prompt_msgs = messages[:-1]
    prompt = _apply(proc, prompt_msgs, True)
    full = _apply(proc, messages, False)
    imgs, vids = process_vision_info(messages)
    full_in = proc(text=[full], images=imgs, videos=vids, return_tensors="pt")
    prompt_in = proc(text=[prompt], images=imgs, videos=vids, return_tensors="pt")
    labels = full_in["input_ids"].clone()
    labels[:, :prompt_in["input_ids"].shape[1]] = -100
    full_in["labels"] = labels
    return {k: v.to(device) for k, v in full_in.items()}


def som_user_content(som, mosaic, extra=""):
    return [
        {"type": "image", "image": som},
        {"type": "image", "image": mosaic},
        {"type": "text", "text": extra or PROMPT_ALL},
    ]


def labels_json(item):
    return {str(i + 1): pub(lab) for i, lab in enumerate(item["labels"])}


def stage_qa(item, kind, example=0):
    labs = item["labels"]
    hint = f"This is worked EXAMPLE {example} with gold labels." if example else ICL_HINT
    if kind == "dedup":
        q = (
            PROMPT + "\n\n" + hint
            + "\nTask 1/4: FULL SoM only. If several boxes sit on the SAME fruit, keep the tighter one, extras dup. "
            + "Different fruits that touch are both keep.\n"
            + 'JSON only: {"1":"keep","2":"dup"}'
        )
        a = {str(i + 1): ("dup" if lab == "dup" else "keep") for i, lab in enumerate(labs)}
        return q, a
    if kind == "filt":
        q = (
            PROMPT + "\n\n" + hint
            + "\nTask 2/4: tomato vs not_tomato. Use SoM+mosaic. Conservative not_tomato if no fruit. "
            + "Also not_tomato if the fruit is tiny vs others in the SoM.\n"
            + 'JSON only: {"1":"tomato","2":"not_tomato"}'
        )
        a = {str(i + 1): ("not_tomato" if lab in ("other", "small") else "tomato")
             for i, lab in enumerate(labs) if lab != "dup"}
        return q, a or {"none": "tomato"}
    if kind == "unripe":
        q = (
            PROMPT + "\n\n" + hint
            + "\nTask 3/4: tomato marks only. unripe vs later. Keep fruit behind another tomato.\n"
            + 'JSON only: {"1":"unripe","3":"later"}'
        )
        a = {str(i + 1): ("unripe" if lab == "unripe" else "later")
             for i, lab in enumerate(labs) if lab in TOMATO}
        return q, a or {"none": "later"}
    q = (
        PROMPT + "\n\n" + hint
        + "\nTask 4/4: leftover fruit ripe vs semi_ripe. Ignore unripe, not_tomato, dup.\n"
        + 'JSON only: {"1":"ripe","2":"semi_ripe"}'
    )
    a = {str(i + 1): lab for i, lab in enumerate(labs) if lab in ("ripe", "semi_ripe")}
    return q, a or {"none": "ripe"}


def vqa_pair(item, rng=None, kind=None):
    rng = rng or random.Random(0)
    kind = kind or rng.choice(("dedup", "filt", "unripe", "ripe_semi"))
    return kind, stage_qa(item, kind)


def icl_stage(exs, kind):
    msgs = []
    for k, ex in enumerate(exs, 1):
        som, mosaic = pair_images(ex)
        q, a = stage_qa(ex, kind, example=k)
        msgs.append({"role": "user", "content": som_user_content(som, mosaic, q)})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(a, ensure_ascii=False)}]})
    return msgs


def json_acc(gold, text):
    try:
        pred = extract_json(text)
    except Exception:
        return 0.0
    if not gold:
        return 0.0

    def nrm(x):
        return str(x).lower().replace("-", "_").strip()

    ok = 0
    for k, v in gold.items():
        p = pred.get(k, pred.get(str(k), ""))
        if nrm(p) == nrm(v):
            ok += 1
    return ok / len(gold)


def pick_exemplars(items, n=2, box_map=None, seed=0):
    rng = random.Random(seed)
    cands = [it for it in items if set(it["labels"]) >= NEED3]
    cands.sort(key=lambda it: -len(it["boxes"]))
    out = []
    for it in cands:
        if box_map is not None:
            ex = label_props(it, box_map.get(it["path"], []))
        else:
            ex = add_distractors(it, k=2, rng=rng)
        n_tom = sum(l in NEED3 for l in ex["labels"])
        if set(ex["labels"]) >= NEED3 and n_tom >= 8:
            cnt = {c: ex["labels"].count(c) for c in CLASSES}
            print(f"ICL {Path(it['path']).name} n={len(ex['boxes'])} gt={len(it['boxes'])} {cnt}")
            out.append(ex)
        if len(out) == n:
            return out
    raise ValueError("need 2 ICL images with ripe+semi_ripe+unripe")


def pick_exemplar(items, seed=0):
    return pick_exemplars(items, n=1, seed=seed)[0]


def icl_messages(exs):
    msgs = []
    for k, ex in enumerate(exs, 1):
        msgs.extend(exemplar_messages(ex, k))
    return msgs


def exemplar_messages(ex, k=1):
    som, mosaic = pair_images(ex)
    ans = labels_json(ex)
    return [
        {"role": "user", "content": som_user_content(som, mosaic, prompt_ex(k) + "\n" + size_note(ex))},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(ans, ensure_ascii=False)}]},
    ]


def with_icl(icl, item, text=None, answer=None):
    som, mosaic = pair_images(item)
    body = (text or PROMPT_ALL) + "\n" + size_note(item)
    msgs = list(icl) + [{"role": "user", "content": som_user_content(som, mosaic, body)}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(answer, ensure_ascii=False)}]})
    return msgs


def match_det(gts, preds, thr=0.5):
    preds = sorted(preds, key=lambda x: x[2], reverse=True)
    used, matched, pairs, fp = set(), set(), [], []
    for box, cls, conf in preds:
        best, biou = -1, thr
        for i, (gb, gc) in enumerate(gts):
            if i in used or gc != cls:
                continue
            v = iou(box, gb)
            if v > biou:
                best, biou = i, v
        if best >= 0:
            used.add(best)
            matched.add(best)
            pairs.append((gts[best][1], cls))
        else:
            fp.append(cls)
    for i, (_, gc) in enumerate(gts):
        if i not in matched:
            pairs.append((gc, "miss"))
    for cls in fp:
        pairs.append(("none", cls))
    return pairs


def det_score(pairs):
    out, f1s = {}, []
    for c in TOMATO:
        tp = sum(g == p == c for g, p in pairs)
        fp = sum(p == c and g != c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        out[c] = {"p": p_, "r": r_, "f1": f1, "n": sum(g == c for g, p in pairs)}
        f1s.append(f1)
    acc = sum(g == p for g, p in pairs) / max(1, len(pairs))
    macro = sum(f1s) / 3
    return {
        "acc": acc, "macro_f1": macro, "tomato_macro_f1": macro,
        "tomato_acc": acc, "other_acc": 0.0, "per_class": out, "n": len(pairs),
    }


def auto_bs(has_icl):
    import torch
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    if has_icl:
        return 2 if free >= 22 else 1
    return 6 if free >= 30 else 4 if free >= 18 else 2


def eval_yolo1_vlm(model, proc, raw_items, props, icl, bs=None, desc="eval", qual=True):
    from tqdm import tqdm
    bs = bs or auto_bs(bool(icl))
    props_items = [label_props(it, props.get(it["path"], [])) for it in raw_items]
    pred_maps = []
    print(f"{desc} bs={bs} icl_turns={len(icl)//2} boxes=yolo1@{PROP_CONF} (not GT)")
    for i in tqdm(range(0, len(props_items), bs), desc=desc):
        chunk = props_items[i:i + bs]
        ntok = min(192, 32 + 8 * max(1, max(len(it["boxes"]) for it in chunk)))
        outs = generate_batch(model, proc, [with_icl(icl, it) for it in chunk], max_new_tokens=ntok)
        if i == 0:
            print("raw[0]:", outs[0][:300])
        for it, text in zip(chunk, outs):
            try:
                pred_maps.append(parse_mark_map(text, len(it["boxes"])))
            except Exception:
                pred_maps.append({})
    pairs = []
    for raw, prop, pred in zip(raw_items, props_items, pred_maps):
        vlm = [
            (prop["boxes"][j], pred.get(j + 1, "other"), 1.0)
            for j in range(len(prop["boxes"])) if pred.get(j + 1) in TOMATO
        ]
        gts = [(b, lab) for b, lab in zip(raw["boxes"], raw["labels"]) if lab in TOMATO]
        pairs.extend(match_det(gts, vlm))
    sc = det_score(pairs)
    if qual:
        try:
            save_qual(desc.replace("-eval", ""), raw_items[:10], props_items[:10], pred_maps[:10])
        except Exception as e:
            print("qual skip", e)
    return sc


def lora_dir(tag):
    for name in (f"{tag}_lora_best", f"{tag}_lora_last", f"{tag}_lora"):
        p = OUT / name
        if p.exists():
            return p
    return None


def ckpt_eval(model, proc, test, y1, icl, tag, step, best, extra=""):
    """Eval 32-img, log, save `{tag}_last` and `{tag}_best`."""
    import torch
    torch.cuda.empty_cache()
    train_on = model.training
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    try:
        sc = eval_yolo1_vlm(model, proc, test, y1, icl, desc=f"{tag}-{step}", qual=False)
    finally:
        if train_on:
            model.train()
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
    f1 = sc.get("macro_f1", sc.get("tomato_macro_f1", 0))
    pc = sc.get("per_class", {})
    line = (
        f"{tag} step={step} acc={sc['acc']:.3f} macroF1={f1:.3f} "
        f"ripe={pc.get('ripe', {}).get('f1', 0):.3f} "
        f"semi={pc.get('semi_ripe', {}).get('f1', 0):.3f} "
        f"unripe={pc.get('unripe', {}).get('f1', 0):.3f} {extra}"
    ).rstrip()
    print(line, flush=True)
    rec = {"step": step, "acc": sc["acc"], "macro_f1": f1, "n": sc.get("n"), "extra": extra}
    with (OUT / f"{tag}.log").open("a") as f:
        f.write(line + "\n")
    with (OUT / f"{tag}_metrics.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    model.save_pretrained(OUT / f"{tag}_lora_last")
    if f1 >= best[0]:
        best[0] = f1
        model.save_pretrained(OUT / f"{tag}_lora_best")
        print(f"  saved best macroF1={f1:.3f}", flush=True)
    return sc


def score(pairs):
    total = len(pairs)
    if not total:
        return {"acc": 0.0, "n": 0}
    acc = sum(g == p for g, p in pairs) / total
    per = {}
    for c in CLASSES:
        tp = sum(g == p == c for g, p in pairs)
        fp = sum(p == c and g != c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"p": prec, "r": rec, "f1": f1, "n": sum(g == c for g, p in pairs)}
    tomato = [(g, p) for g, p in pairs if g != "other"]
    other = [(g, p) for g, p in pairs if g == "other"]
    tf1 = [per[c]["f1"] for c in TOMATO]
    return {
        "acc": acc,
        "n": total,
        "tomato_acc": sum(g == p for g, p in tomato) / len(tomato) if tomato else 0.0,
        "other_acc": sum(g == p for g, p in other) / len(other) if other else 0.0,
        "tomato_macro_f1": sum(tf1) / len(tf1),
        "per_class": per,
    }


def _f1s(m):
    pc = m.get("per_class", {})
    return {c: pc.get(c, {}).get("f1", 0.0) for c in TOMATO}


def report_vs_yolo(name, metrics):
    y32 = json.loads((OUT / "yolo32.json").read_text()) if (OUT / "yolo32.json").exists() else None
    y161 = None
    p161 = OUT / "yolo_vs_rl.json"
    if p161.exists():
        y161 = json.loads(p161.read_text()).get("yolo")

    def line(tag, m, note=""):
        f = _f1s(m)
        macro = m.get("macro_f1", m.get("tomato_macro_f1", sum(f.values()) / 3))
        return (
            f"{tag:18s} acc={m.get('acc', 0):.3f} tomato={m.get('tomato_acc', m.get('acc', 0)):.3f} "
            f"macroF1={macro:.3f} ripe={f['ripe']:.3f} semi={f['semi_ripe']:.3f} "
            f"unripe={f['unripe']:.3f} {note}"
        ).rstrip()

    print(f"\n===== {name} vs YOLO 3cls =====")
    if y32:
        print(line("YOLO 3cls (32)", y32, "[det IoU0.5]"))
    if y161:
        print(line("YOLO 3cls (161)", y161, "[det IoU0.5 full]"))
    print(line(name, metrics, "[SoM cls 32img]"))
    print("================================\n")


def dump(name, metrics, extras=None):
    payload = {"name": name, **metrics}
    if extras:
        payload.update(extras)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[{name}] acc={metrics['acc']:.3f} tomato={metrics.get('tomato_acc', metrics['acc']):.3f} "
          f"macroF1={metrics.get('macro_f1', metrics.get('tomato_macro_f1', 0)):.3f} n={metrics['n']}")
    report_vs_yolo(name, metrics)
    return path


def paint_cls(item, dets, orig, w=480):
    im = orig.resize((w, int(orig.size[1] * w / orig.size[0])), Image.BILINEAR)
    sx, sy = im.size[0] / item["wh"][0], im.size[1] / item["wh"][1]
    d = ImageDraw.Draw(im)
    for box, cls in dets:
        x1, y1, x2, y2 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
        c = CLS_COL.get(cls, (160, 160, 160))
        d.rectangle([x1, y1, x2, y2], outline=c, width=5)
        if cls != "yolo1":
            tag = {"ripe": "R", "semi_ripe": "S", "unripe": "U", "not_tomato": "X"}.get(cls, cls)
            _badge(d, (max(2, x1), max(2, y1)), tag, c, QUAL_FONT)
    return im


def _panel(im, title):
    top = 36
    canvas = Image.new("RGB", (im.size[0], im.size[1] + top), (18, 20, 22))
    ImageDraw.Draw(canvas).text((8, 6), title, fill=(255, 255, 255), font=FONT)
    canvas.paste(im, (0, top))
    return canvas


def save_qual(name, raws, props, preds, n=10):
    from ultralytics import YOLO
    dst = OUT / "qual" / name
    dst.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(YOLO_3CLS_W)
    html = [f"<html><body><h1>{name}</h1>",
            "<p>GT | YOLO 3cls | YOLO 1cls@0.1 | ours (not_tomato dropped). "
            "red=ripe orange=semi green=unripe cyan=1cls</p>"]
    n = min(n, len(raws), len(props), len(preds))
    for i in range(n):
        raw, prop, pred = raws[i], props[i], preds[i]
        orig = open_rgb(raw["path"])
        gt = [(b, l) for b, l in zip(raw["boxes"], raw["labels"]) if l in TOMATO]
        y1b = [(b, "yolo1") for b in prop["boxes"]]
        ours = [(prop["boxes"][j], pred.get(j + 1))
                for j in range(len(prop["boxes"])) if pred.get(j + 1) in TOMATO]
        y3 = detect_cls(yolo, raw, YOLO3_NAMES)
        pans = [
            _panel(paint_cls(raw, gt, orig), "GT"),
            _panel(paint_cls(raw, y3, orig), "YOLO 3cls"),
            _panel(paint_cls(raw, y1b, orig), "YOLO 1cls"),
            _panel(paint_cls(raw, ours, orig), name),
        ]
        row = Image.new("RGB", (sum(p.size[0] for p in pans), max(p.size[1] for p in pans)), (18, 20, 22))
        x = 0
        for p in pans:
            row.paste(p, (x, 0))
            x += p.size[0]
        row.save(dst / f"{i:02d}.jpg", quality=90)
        html.append(f"<h3>{i:02d} {Path(raw['path']).name}</h3><img src='{i:02d}.jpg' width=1920>")
    (dst / "index.html").write_text("".join(html) + "</body></html>")
    write_qual_index()
    print(f"qual {dst} n={n}")
    del yolo


def write_qual_index():
    dst = OUT / "qual"
    dst.mkdir(parents=True, exist_ok=True)
    html = ["<html><body><h1>YOLO 3cls vs VLM</h1>",
            "<p>red=ripe, orange=semi, green=unripe</p>"]
    for d in sorted(p for p in dst.iterdir() if p.is_dir()):
        jpgs = sorted(d.glob("*.jpg"))
        if not jpgs:
            continue
        html.append(f"<h2>{d.name}</h2>")
        for p in jpgs:
            html.append(f"<p>{p.stem}</p><img src='{d.name}/{p.name}' width=1920>")
    (dst / "index.html").write_text("".join(html) + "</body></html>")
