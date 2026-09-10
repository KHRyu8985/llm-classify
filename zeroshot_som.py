"""Zero-shot / ICL SoM on YOLO 1-class proposals (conf=0.1)."""
import argparse
from som_common import (
    set_dataset, prepare, load_vlm, dump, collect_yolo1, pick_exemplars,
    icl_messages, eval_yolo1_vlm,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=32)
    ap.add_argument("--icl", action="store_true")
    ap.add_argument("--ds", default="little")
    args = ap.parse_args()
    set_dataset(args.ds)
    train, test = prepare(max_eval=args.max_eval)
    props = collect_yolo1(train + test)
    icl = icl_messages(pick_exemplars(train, n=2, box_map=props)) if args.icl else []
    name = "icl" if args.icl else "zeroshot"
    model, proc = load_vlm()
    dump(name, eval_yolo1_vlm(model, proc, test, props, icl, desc=name),
         extras={"protocol": "yolo1@0.1 + vlm", "ds": args.ds})


if __name__ == "__main__":
    main()
