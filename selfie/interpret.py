from selfie.generate_wrappers import generate_interpret, model_forward_interpret
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

class InterpretationPrompt:
    def __init__(self, tokenizer, interpretation_prompt_sequence):
        self.tokenizer = tokenizer

        self.interpretation_prompt = ""
        self.insert_locations = []

        for part in interpretation_prompt_sequence:
            # for every element in tuple check if string or not
            if type(part) == str:
                # if element in tuple is string, add to interpretation prompt
                self.interpretation_prompt += part
            else:
                # Calculate starting position for tokens insertion:
                # `self.tokenizer.encode()` converts  current interpretation_promp into token IDs
                # `len(...)` gives the number of tokens so far.
                insert_start = len(self.tokenizer.encode(self.interpretation_prompt))
                # Modify interpretation_prompt by appending "_ " (placeholder).
                # changes the content of the prompt and will likely change the tokenization results.
                self.interpretation_prompt += "_ "
                # Calculate  ending position after the new addition:
                # re-encoding updated interpretation_prompt: total number of tokens now present are shown.
                insert_end = len(self.tokenizer.encode(self.interpretation_prompt))
                # Iterate over the range of indices corresponding to the tokens added by "_ ".
                # add each index from to insert_locations list (positions of the new tokens)
                # in the tokenizer's tokenized version of the updated `interpretation_prompt`.
                for insert_idx in range(insert_start, insert_end):
                    self.insert_locations.append(insert_idx)
        # tokenize prompt
        self.interpretation_prompt_model_inputs = self.tokenizer(self.interpretation_prompt, return_tensors="pt")
            




def interpret(original_prompt = None,
              tokenizer = None,
                interpretation_prompt = None,
                model = None,
                tokens_to_interpret = None,
                bs = 8,
                max_new_tokens = 30,
                k = 1):
    # ADDED:
    # set device to cpu (in original code everything was done on GPU but our resources are limited)
    device = torch.device("cpu")
    model.to(device)
    #
    # tokenized prompt with placeholders "_ "
    interpretation_prompt_model_inputs = interpretation_prompt.interpretation_prompt_model_inputs
    # list of positions with placeholders
    insert_locations = interpretation_prompt.insert_locations

    # tokenize original prompt & move original and interpretation prompt tensors to cpu for processing
    # changed .to(model.device) to .to("cpu")
    original_prompt_inputs = tokenizer(original_prompt, return_tensors="pt").to(model.device)
    interpretation_prompt_model_inputs = interpretation_prompt_model_inputs.to(model.device)

    # prep interpretation df
    interpretation_df = {
        'prompt': [],
        'interpretation': [],
        'layer': [],
        'token': [],
        'token_decoded': [],
    }

    # prompt len is not used, but gets the tensor length from the original prompt
    # ([-1] selects last dimension which is the number of tokens)
    prompt_len = original_prompt_inputs['input_ids'].shape[-1]
    outputs = model_forward_interpret(model,
                **original_prompt_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=True,
            )
    
    #print(tokenizer.decode(outputs))
    # ADDED:
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    #
    
    # assign everything to insert_info
    all_insert_infos = []
    for retrieve_layer, retrieve_token in tokens_to_interpret:
            insert_info = {}
            insert_info['replacing_mode'] = 'normalized'
            insert_info['overlay_strength'] = 1
            insert_info['retrieve_layer'] = retrieve_layer
            insert_info['retrieve_token'] = retrieve_token
            # for each layer in the model check if the layer is layer k (for insertion)
            for layer_idx, layer in enumerate(model.model.layers):
                if layer_idx == k:
                    insert_locations = interpretation_prompt.insert_locations
                    # for layer k: identify hidden states (in the first batch) corresponding to the the specified layer
                    # & repeat duplicates of the retrieved token's hidden state across multiple positions (corresponding to insert_locations).
                    # Result: A tensor with shape (1, len(insert_locations), hidden_size) where the hidden state of the retrieve_token is repeated for every index in insert_locations
                    insert_info[layer_idx] = (insert_locations, outputs['hidden_states'][retrieve_layer][0][retrieve_token].repeat(1,len(insert_locations), 1))
            all_insert_infos.append(insert_info)
    
    # ADDED:
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    #

    # Iterate through `all_insert_infos` in batches of size `bs` (batch size)
    for batch_start_idx in tqdm(range(0,len(all_insert_infos),bs)):
        with torch.no_grad(): # Disable gradient calculation for faster computation (inference)
            # Extract the current batch of insert information
            # slice of the `all_insert_infos` list starting at batch_start_idx and ending at either batch_start_idx + bs
            # or the end of the list (whichever comes first)
            batch_insert_infos = all_insert_infos[batch_start_idx:min(batch_start_idx+bs, len(all_insert_infos))]
            # define number of tokens in the interpretation prompt to determine where the
            # generated output begins after the static prompt tokens.
            repeat_prompt_n_tokens = interpretation_prompt_model_inputs['input_ids'].shape[-1]
            # added .to("cpu") at the end
            # Prepare batched inputs for the interpretation prompt:
            # encode the interpretation_prompt repeated len(batch_insert_infos) times (once for each item in the current batch).
            batched_interpretation_prompt_model_inputs = tokenizer([interpretation_prompt.interpretation_prompt] * len(batch_insert_infos), return_tensors="pt").to("cpu")
            # Call the generate_interpret function to generate outputs for the current batch
            output = generate_interpret(**batched_interpretation_prompt_model_inputs, model=model, max_new_tokens=max_new_tokens, insert_info=batch_insert_infos, pad_token_id=tokenizer.eos_token_id, output_attentions = False)
            
            cropped_interpretation_tokens = output[:,repeat_prompt_n_tokens:]
            cropped_interpretation = tokenizer.batch_decode(cropped_interpretation_tokens, skip_special_tokens=True)

            for i in range(len(batch_insert_infos)):
                interpretation_df['prompt'].append(original_prompt)
                interpretation_df['interpretation'].append(cropped_interpretation[i])
                interpretation_df['layer'].append(batch_insert_infos[i]['retrieve_layer'])
                interpretation_df['token'].append(batch_insert_infos[i]['retrieve_token'])
                interpretation_df['token_decoded'].append(tokenizer.decode(original_prompt_inputs.input_ids[0, batch_insert_infos[i]['retrieve_token']]))
    
    # ADDED:
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    #
    
    return interpretation_df

    
    
def interpret_vectors(vecs=None, model=None, interpretation_prompt=None, tokenizer=None, bs = 8, k = 2, max_new_tokens=30):
    interpretation_prompt_model_inputs = interpretation_prompt.interpretation_prompt_model_inputs
    insert_locations = interpretation_prompt.insert_locations
    # replaced .to(model.device) with .to("cpu")
    interpretation_prompt_model_inputs = interpretation_prompt_model_inputs.to("cpu")

    all_interpretations = []

    batch_insert_infos = []

    batch_insert_infos = []

    for vec_idx, vec in enumerate(vecs):
        insert_info = {}
        insert_info['replacing_mode'] = 'normalized'
        insert_info['overlay_strength'] = 1

        # insert_info['replacing_mode'] = 'addition'
        # insert_info['overlay_strength'] = 1000

        insert_info[1] = (insert_locations, vec.repeat(1,len(insert_locations), 1))

        batch_insert_infos.append(insert_info)

        if len(batch_insert_infos) == bs or vec_idx == len(vecs) - 1:
            # replaced .to('cuda:0') with .to("cpu")
            batched_interpretation_prompt_model_inputs = tokenizer([interpretation_prompt.interpretation_prompt] * len(batch_insert_infos), return_tensors="pt").to("cpu")
            repeat_prompt_n_tokens = interpretation_prompt_model_inputs['input_ids'].shape[-1]
            output = generate_interpret(**batched_interpretation_prompt_model_inputs, model=model, max_new_tokens=max_new_tokens, insert_info=batch_insert_infos, pad_token_id=tokenizer.eos_token_id, output_attentions = False)
            
            cropped_interpretation_tokens = output[:,repeat_prompt_n_tokens:]
            cropped_interpretation = tokenizer.batch_decode(cropped_interpretation_tokens, skip_special_tokens=True)
            all_interpretations.extend(cropped_interpretation)
            batch_insert_infos = []

    return all_interpretations