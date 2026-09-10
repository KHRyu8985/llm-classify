"""Train YOLO11 1-class or 3-class. Copies best.pt into weights/{ds}/."""
import argparse, shutil
from pathlib import Path
from ultralytics import YOLO

DATA = {
    ("little", "3"): "data/yolo/laboro_little_yolo_3cls/data.yaml",
    ("little", "1"): "data/yolo/laboro_little_yolo_1cls/data.yaml",
    ("kaggle", "3"): "data/yolo/laboro_kaggle_3cls/data.yaml",
    ("kaggle", "1"): "data/yolo/laboro_big_yolo_1cls/data.yaml",
    ("laboro_tomatod", "3"): "data/yolo/laboro_tomatod_yolo_3cls/data.yaml",
    ("laboro_tomatod", "1"): "data/yolo/laboro_tomatod_yolo_1cls/data.yaml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="little")
    ap.add_argument("--cls", choices=("1", "3"), default="3")
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    name = "yolo11_1cls" if args.cls == "1" else "yolo11"
    dst = Path("weights") / args.ds
    dst.mkdir(parents=True, exist_ok=True)
    YOLO("yolo11m.pt").train(
        data=DATA[(args.ds, args.cls)], epochs=args.epochs, imgsz=640, batch=12,
        project=str(Path("runs/detect") / args.ds), name=name, exist_ok=True,
    )
    src = Path("runs/detect") / args.ds / name / "weights" / "best.pt"
    out = dst / f"{name}.pt"
    shutil.copy2(src, out)
    print("saved", out)


if __name__ == "__main__":
    main()
