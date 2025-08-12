import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from goldentransformer.core.fault_injector import FaultInjector
from goldentransformer.faults.weight_corruption import WeightCorruption
import numpy as np
import random

prompt = "The meaning of life is"
model_name = "gpt2"
layer_to_corrupt = [0, 5, 10]
corruption_rate = 1e-12
seed = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate(model, tokenizer, prompt, max_new_tokens=20):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    # Uncorrupted model
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("\n[Uncorrupted Model Output]")
    print(generate(model, tokenizer, prompt, max_new_tokens=50))
    # Corrupted model
    set_seed(seed)
    model_corrupt = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    injector = FaultInjector(model_corrupt)
    fault = WeightCorruption(
        pattern="bit_flip",
        corruption_rate=corruption_rate,
        target_layers=layer_to_corrupt
    )
    injector.inject_fault(fault)
    print("\n[Bit-Flip Corrupted Model Output]")
    print(generate(model_corrupt, tokenizer, prompt, max_new_tokens=50))

if __name__ == "__main__":
    main() 