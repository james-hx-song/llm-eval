from wrapper import LMWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import lm_eval
import json


device = 'cuda' if torch.cuda.is_available() else 'cpu'

instruct = True
model_name = "meta-llama/Llama-3.2-1B" if not instruct else "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

batch_size = 4

# Using my wrapper (can be used on any model)
myLM = LMWrapper(model, tokenizer, batch_size, device=device, model_type="huggingface")

# Using HuggingfaceLM (can only be used on Huggingface models)
# myLM = lm_eval.models.huggingface.HFLM(
#     pretrained="meta-llama/Llama-3.2-1B-Instruct",
#     tokenizer="meta-llama/Llama-3.2-1B-Instruct",
#     device=device
# )

task_manager = lm_eval.tasks.TaskManager()

task = "gsm8k"
results = lm_eval.simple_evaluate( # call simple_evaluate
    model=myLM,
    tasks=[task],
    num_fewshot=8,
    task_manager=task_manager,
    device=device
)
file = f"instruct_results_{task}.json" if instruct else f"results_{task}.json"
with open(file, "w") as f:
    json.dump(results, f, indent=4)

