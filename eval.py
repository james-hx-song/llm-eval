from wrapper import LMWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import lm_eval
import json


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

batch_size = 4

# Using my wrapper (can be used on any model)
myLM = LMWrapper(model, tokenizer, batch_size, device=device)

# Using HuggingfaceLM (can only be used on Huggingface models)
# myLM = lm_eval.models.huggingface.HFLM(
#     pretrained="meta-llama/Llama-3.2-1B-Instruct",
#     tokenizer="meta-llama/Llama-3.2-1B-Instruct",
#     device=device
# )

task_manager = lm_eval.tasks.TaskManager()

task = "arc_challenge"
results = lm_eval.simple_evaluate( # call simple_evaluate
    model=myLM,
    tasks=[task],
    num_fewshot=25,
    task_manager=task_manager,
    device=device
)

with open(f"instruct_results_{task}.json", "w") as f:
    json.dump(results, f, indent=4)

