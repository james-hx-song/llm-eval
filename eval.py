from wrapper import LMWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import lm_eval
import json


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

batch_size = 2
myLM = LMWrapper(model, tokenizer, batch_size, device=device)

task_manager = lm_eval.tasks.TaskManager()

task = "hellaswag"
results = lm_eval.simple_evaluate( # call simple_evaluate
    model=myLM,
    tasks=[task],
    num_fewshot=5,
    task_manager=task_manager,
    device=device
)

with open(f"instr_results_{task}.json", "w") as f:
    json.dump(results, f, indent=4)

