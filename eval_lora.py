"""Re-eval saved SFT or RL LoRA. Does not train."""
import argparse
from som_common import (
    set_dataset, prepare, load_vlm, dump, collect_yolo1, pick_exemplars,
    icl_messages, eval_yolo1_vlm, lora_dir,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="little")
    ap.add_argument("--tag", default="sft", choices=("sft", "rl"))
    ap.add_argument("--max-eval", type=int, default=32)
    args = ap.parse_args()
    set_dataset(args.ds)
    from som_common import OUT
    train, test = prepare(max_eval=args.max_eval)
    y1 = collect_yolo1(train + test)
    icl = icl_messages(pick_exemplars(train, n=2, box_map=y1))
    path = lora_dir(args.tag) or (OUT / f"{args.tag}_lora")
    model, proc = load_vlm(merge_lora=str(path))
    dump(args.tag, eval_yolo1_vlm(model, proc, test, y1, icl, desc=args.tag),
         extras={"protocol": "yolo1@0.1 + vlm", "ds": args.ds, "lora": str(path)})


if __name__ == "__main__":
    main()
