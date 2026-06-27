import os, json, csv, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
IN = os.environ.get("INPUT_PATH", "/data/public_test.json")
OUT = os.environ.get("OUTPUT_PATH", "/data/pred.csv")
LETTERS = ["A","B","C","D","E","F"]

def build_prompt(q, choices):
    s = f"Câu hỏi: {q}\n"
    for i,c in enumerate(choices):
        s += f"{LETTERS[i]}. {c}\n"
    s += "Chỉ trả lời bằng 1 chữ cái. Đáp án:"
    return s

def extract(text, n):
    m = re.search(r"\b([A-F])\b", text.upper())
    if m and LETTERS.index(m.group(1)) < n:
        return m.group(1)
    return "A"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
    data = json.load(open(IN, encoding="utf-8"))
    rows = [("qid","answer")]
    for i, item in enumerate(data):
        qid = item.get("id", item.get("qid", str(i)))
        q = item.get("question","")
        choices = item.get("choices") or item.get("options") or []
        msgs = [{"role":"user","content": build_prompt(q, choices)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=8, do_sample=False)
        txt = tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        rows.append((qid, extract(txt, len(choices))))
        if i % 20 == 0: print(f"Done {i}/{len(data)}", flush=True)
    with open(OUT,"w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print("DONE:", OUT)

if __name__ == "__main__":
    main()

