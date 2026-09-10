"""GRPO-lite RL with ICL + expert prompt."""
import argparse, random
from tqdm import tqdm
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from som_common import (
    set_dataset, prepare, load_vlm, generate, dump, encode_sft, ckpt_eval,
    collect_yolo1, label_props, shuffle_marks,
    pick_exemplars, icl_messages, vqa_pair, icl_stage, with_icl, eval_yolo1_vlm, json_acc, lora_dir,
)


def sample_logprob(model, proc, item, exs, rng, temperature=1.0):
    kind, (q, a) = vqa_pair(item, rng)
    icl = icl_stage(exs[:1], kind)
    model.eval()
    model.gradient_checkpointing_disable()
    try:
        text, inputs, out = generate(
            model, proc, with_icl(icl, item, text=q), max_new_tokens=192, do_sample=True, temperature=temperature
        )
    finally:
        model.train()
        model.gradient_checkpointing_enable()
    plen = inputs["input_ids"].shape[1]
    labels = out.clone()
    labels[:, :plen] = -100
    vision = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask", "labels")}
    with torch.enable_grad():
        loss = model(input_ids=out, attention_mask=torch.ones_like(out), labels=labels, **vision).loss
    return text, -loss, q, a, kind


def train_rl(model, proc, items, exs, test, y1, icl, steps, k=2, lr=5e-5):
    from som_common import OUT
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    best, sc = [-1.0], None
    log = (OUT / "rl.log").open("a")
    for step in tqdm(range(steps), desc="rl"):
        rng = random.Random(step)
        item = shuffle_marks(items[step % len(items)], rng)
        logps, rs = [], []
        q, a, kind = None, None, None
        for _ in range(k):
            t, lp, q, a, kind = sample_logprob(model, proc, item, exs, rng, temperature=1.0)
            logps.append(lp)
            rs.append(json_acc(a, t))
        r = torch.tensor(rs, device=model.device, dtype=torch.float32)
        loss = -((r - r.mean()).detach() * torch.stack(logps)).mean()
        loss = loss + 0.5 * model(**encode_sft(proc, with_icl(icl_stage(exs[:1], kind), item, text=q, answer=a), model.device)).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        extra = f"{kind} R={sum(rs)/k:.3f} loss={loss.item():.4f}"
        if (step + 1) % 20 == 0 or step + 1 == steps:
            print(f"step {step + 1} {extra}", flush=True)
            log.write(f"step {step + 1} {extra}\n")
            log.flush()
            sc = ckpt_eval(model, proc, test, y1, icl, "rl", step + 1, best, extra=extra)
    log.close()
    model.eval()
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ds", default="little")
    args = ap.parse_args()
    set_dataset(args.ds)
    train_raw, test_raw = prepare(max_eval=args.max_eval, max_train=args.max_train)
    y1 = collect_yolo1(train_raw + test_raw)
    train_items = [it for it in (label_props(x, y1.get(x["path"], [])) for x in train_raw) if it["boxes"]]
    exs = pick_exemplars(train_raw, n=2, box_map=y1)
    icl = icl_messages(exs)
    model, proc = load_vlm(device=args.device)
    sft_dir = lora_dir("sft")
    if sft_dir:
        model = PeftModel.from_pretrained(model, str(sft_dir), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    k = 4 if free >= 28 else 3 if free >= 16 else 2
    print(f"RL k={k} free={free:.1f}GiB temp=1.0")
    sc = train_rl(model, proc, train_items, exs, test_raw, y1, icl, args.steps, k=k)
    dump("rl", sc or eval_yolo1_vlm(model, proc, test_raw, y1, icl, desc="rl-eval"),
         extras={"protocol": "yolo1@0.1 + vlm", "ds": args.ds})


if __name__ == "__main__":
    main()
