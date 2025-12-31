from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "allenai/OLMo-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

# Load dataset
print("Loading BLIMP dataset...")
ds = load_dataset("nyu-mll/blimp", "adjunct_island", split="train")


def tokenize(sent):
    # OLMo doesn't use token_type_ids, so we exclude them
    encoding = tokenizer(sent, return_tensors="pt", add_special_tokens=True)
    # Remove token_type_ids if present
    if "token_type_ids" in encoding:
        del encoding["token_type_ids"]
    return encoding.to(device)


def get_logprob(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        seq_len = inputs["input_ids"].shape[1]
        log_prob = -outputs.loss.item() * (seq_len - 1)
    return log_prob


correct_count = 0
total = 0

for row in tqdm(ds):
    sent_good = row["sentence_good"]
    sent_bad = row["sentence_bad"]

    good_lp = get_logprob(model, tokenize(sent_good))
    bad_lp = get_logprob(model, tokenize(sent_bad))

    if good_lp > bad_lp:
        correct_count += 1
    total += 1

print(f"Accuracy: {correct_count}/{total} = {correct_count / total:.2%}")
