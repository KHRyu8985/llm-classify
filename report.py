"""Print YOLO / zeroshot / ICL / SFT / RL table from saved jsons."""
import argparse, json
from som_common import set_dataset

ROWS = (
    ("yolo32", "YOLO 3cls"),
    ("zeroshot", "Zero-shot"),
    ("icl", "ICL"),
    ("sft", "SFT"),
    ("rl", "SFT+RL"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="little")
    args = ap.parse_args()
    set_dataset(args.ds)
    from som_common import OUT
    print(f"{'method':<12} {'acc':>6} {'macroF1':>8} {'ripe':>6} {'semi':>6} {'unripe':>6} {'n':>5}")
    for key, name in ROWS:
        p = OUT / f"{key}.json"
        if not p.exists():
            print(f"{name:<12} missing {p}")
            continue
        d = json.loads(p.read_text())
        pc = d.get("per_class", {})
        print(f"{name:<12} {d['acc']:6.3f} {d['macro_f1']:8.3f} "
              f"{pc['ripe']['f1']:6.3f} {pc['semi_ripe']['f1']:6.3f} "
              f"{pc['unripe']['f1']:6.3f} {d['n']:5d}")


if __name__ == "__main__":
    main()
