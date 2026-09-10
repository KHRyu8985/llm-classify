"""LoRA SFT: ICL + mixed VQA on SoM pairs."""
import argparse, random
from tqdm import tqdm
import torch
from peft import LoraConfig, get_peft_model

from som_common import (
    set_dataset, prepare, load_vlm, dump, encode_sft, ckpt_eval,
    collect_yolo1, label_props, shuffle_marks,
    pick_exemplars, icl_messages, vqa_pair, icl_stage, with_icl, eval_yolo1_vlm,
)


def make_msgs(item, exs, rng):
    kind, (q, a) = vqa_pair(item, rng)
    return with_icl(icl_stage(exs[:1], kind), item, text=q, answer=a)


def train(model, proc, items, exs, test, y1, icl, steps, lr=1e-4):
    from som_common import OUT
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    best, sc = [-1.0], None
    log = (OUT / "sft.log").open("a")
    for step in tqdm(range(steps), desc="sft"):
        rng = random.Random(step)
        item = shuffle_marks(items[step % len(items)], rng)
        loss = model(**encode_sft(proc, make_msgs(item, exs, rng), model.device)).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (step + 1) % 20 == 0 or step + 1 == steps:
            msg = f"step {step + 1} loss={loss.item():.4f}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            sc = ckpt_eval(model, proc, test, y1, icl, "sft", step + 1, best,
                           extra=f"loss={loss.item():.4f}")
    log.close()
    model.eval()
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ds", default="little")
    args = ap.parse_args()
    set_dataset(args.ds)
    train_raw, test_raw = prepare(max_eval=args.max_eval, max_train=args.max_train)
    y1 = collect_yolo1(train_raw + test_raw)
    train_items = [label_props(it, y1.get(it["path"], [])) for it in train_raw]
    train_items = [it for it in train_items if it["boxes"]]
    exs = pick_exemplars(train_raw, n=2, box_map=y1)
    icl = icl_messages(exs)
    print(f"train imgs={len(train_items)} boxes=yolo1@0.1 stages=dedup/filt/unripe/ripe_semi")
    model, proc = load_vlm(device=args.device)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    sc = train(model, proc, train_items, exs, test_raw, y1, icl, args.steps)
    dump("sft", sc or eval_yolo1_vlm(model, proc, test_raw, y1, icl, desc="sft-eval"),
         extras={"protocol": "yolo1@0.1 + vlm", "ds": args.ds})


if __name__ == "__main__":
    main()
