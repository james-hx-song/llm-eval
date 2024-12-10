import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class LMWrapper(LM):
    def __init__(self, model, tokenizer, batch_size,device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
    
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        '''
        Returns a list of log probabilities for each request. 
        '''

        results = []

        for idx in tqdm(range(0, len(requests) + self.batch_size, self.batch_size)):
            batch_requests = requests[idx : min(idx + self.batch_size, len(requests))]

            batch_input_strs = [self.tokenizer.bos_token for request in batch_requests]
            batch_target_strs = [request.args[0] for request in batch_requests]

            results.extend(self.loglik_helper(batch_input_strs, batch_target_strs))

        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        '''
        Returns a list of tuples (log_prob, is_greedy) for each request.

        args:
            requests: list[Instance], where each instance = (input_str, target_str)

        returns:
            list[tuple(float, bool)], where each tuple = (log_prob, is_greedy). 
            
            log_prob is log p(target_str | input_str)
            is_greedy is True if the target_str is the most likely next sequence given the input_str
        '''

        results = []

        for idx in tqdm(range(0, len(requests) + self.batch_size, self.batch_size)):
            batch_requests = requests[idx : min(idx + self.batch_size, len(requests))]

            batch_input_strs = [request.args[0] for request in batch_requests]
            batch_target_strs = [request.args[1] for request in batch_requests]

            results.extend(self.loglik_helper(batch_input_strs, batch_target_strs))
            # print(f"finished batch {idx}, {len(results)} results")

        return results
    
    def generate_until(self, requests: list[Instance]) -> list[str]:
        pass
        '''
        def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []

        print(f"Generating {len(requests)} requests")
        for i, request in enumerate(requests):
            input_str, gen_args = request.args

            # print(input_str)
            # print(gen_args)

            


            input_ids = self.tokenizer.encode(input_str)

            print(gen_args['until'])

            end_token_list = [self.tokenizer.encode(token) for token in gen_args['until']]
            
            max_len = gen_args.get('max_len', 400)

            print(end_token_list)
            
            output_ids = self.model.generate(input_ids,  max_len=max_len, end_token_list=end_token_list, device=self.device)

            output_str = self.tokenizer.decode(output_ids)
            
            print(output_str)

            sys.exit(0)

            results.append(output_str)
            print(f"Processed request {i+1}")
        # results = [" blah " for _ in requests]
        return results
        '''
        return []

    def loglik_helper(
        self, 
        batch_inputs: list[str], 
        batch_targets: list[str]
    ) -> list[tuple[float, bool]]:
        '''
        Helper function to compute log likelihood for a batch of input and target strings.
        '''
        results = []

        if len(batch_inputs) == 0:
            return results

        batch_seq = [batch_inputs[i] + batch_targets[i] for i in range(len(batch_inputs))]
        # print(batch_seq)
        # Combined Sequence
        combined_tokens = self.tokenizer(batch_seq, 
                                        return_tensors="pt", 
                                        padding=True,
                                        add_special_tokens=False)

        combined_mask = combined_tokens["attention_mask"].bool().to(self.device)
        combined_tokens = combined_tokens["input_ids"].to(self.device)
        
        # Input Sequence
        input_tokens = self.tokenizer(batch_inputs,
                                        return_tensors="pt", 
                                        padding=True,
                                        add_special_tokens=False)
        input_len = input_tokens["attention_mask"].sum(dim=1).to(self.device)
        input_tokens = input_tokens["input_ids"].to(self.device)

        # Target Sequence
        target_tokens = self.tokenizer(batch_targets,
                                        return_tensors="pt", 
                                        padding=True,
                                        add_special_tokens=False)
        # target_len = target_tokens["attention_mask"].sum(dim=1)
        target_mask = target_tokens["attention_mask"].to(self.device)
        target_tokens = target_tokens["input_ids"].to(self.device)

        # print(target_tokens.shape)

        # Pass Sequence into Model
        logits = self.model(combined_tokens, return_dict=True,).logits
        log_probs = F.log_softmax(logits, dim=-1)
        # Loop through each sequence in the batch
        for batch_idx in range(logits.shape[0]):
            curr_log_probs = log_probs[batch_idx]
            curr_target_tokens = torch.masked_select(target_tokens[batch_idx], target_mask.bool()[batch_idx])

            # Get Target Sequence Length
            target_len = target_mask[batch_idx].sum().item()

            # Get Start Index
            start_idx = input_len[batch_idx] - 1

            # Get Logits for Target Sequence
            curr_log_probs = curr_log_probs[start_idx:start_idx + target_len, :]

            # Get probabilities for target sequence
            row_idx = torch.arange(target_len).to(self.device)

            # print("Target Length: ", target_len)

            log_proba = curr_log_probs[row_idx, curr_target_tokens]

            # Get Greedy Predictions
            greedy_preds = torch.argmax(curr_log_probs, dim=-1)
            is_greedy = torch.all(curr_target_tokens == greedy_preds)

            results.append((log_proba.sum().item(), is_greedy.item()))

        return results



