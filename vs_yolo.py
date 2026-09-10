"""YOLO 3-class F1 on the same 32-image VLM split."""
import argparse, json
from tqdm import tqdm
from ultralytics import YOLO
import som_common as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="little")
    ap.add_argument("--max-eval", type=int, default=32)
    args = ap.parse_args()
    sc.set_dataset(args.ds)
    items = sc.load_split("test")[:args.max_eval]
    yolo = YOLO(sc.YOLO_3CLS_W)
    pairs = []
    for it in tqdm(items, desc="yolo3"):
        preds = [(b, c, 1.0) for b, c in sc.detect_cls(yolo, it, sc.YOLO3_NAMES)]
        gts = [(b, lab) for b, lab in zip(it["boxes"], it["labels"]) if lab in sc.TOMATO]
        pairs.extend(sc.match_det(gts, preds))
    out = {"name": "yolo3cls_32", **sc.det_score(pairs), "ds": args.ds}
    (sc.OUT / "yolo32.json").write_text(json.dumps(out, indent=2))
    pc = out["per_class"]
    print(f"YOLO32 acc={out['acc']:.3f} macroF1={out['macro_f1']:.3f} "
          f"ripe={pc['ripe']['f1']:.3f} semi={pc['semi_ripe']['f1']:.3f} "
          f"unripe={pc['unripe']['f1']:.3f} n={out['n']}")


if __name__ == "__main__":
    main()
