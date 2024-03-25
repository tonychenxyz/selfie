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
            if type(part) == str:
                self.interpretation_prompt += part
            else:
                insert_start = len(self.tokenizer.encode(self.interpretation_prompt))
                self.interpretation_prompt += "_ "
                insert_end = len(self.tokenizer.encode(self.interpretation_prompt))

                for insert_idx in range(insert_start, insert_end):
                    self.insert_locations.append(insert_idx)
        
        self.interpretation_prompt_model_inputs = self.tokenizer(self.interpretation_prompt, return_tensors="pt")
            




def interpret(original_prompt = None,
              tokenizer = None,
                interpretation_prompt = None,
                model = None,
                tokens_to_interpret = None,
                bs = 8,
                max_new_tokens = 30,
                k = 1):

    print(f"Interpreting '{original_prompt}' with '{interpretation_prompt.interpretation_prompt}'")
    interpretation_prompt_model_inputs = interpretation_prompt.interpretation_prompt_model_inputs
    insert_locations = interpretation_prompt.insert_locations
    original_prompt_inputs = tokenizer(original_prompt, return_tensors="pt").to(model.device)
    interpretation_prompt_model_inputs = interpretation_prompt_model_inputs.to(model.device)

    interpretation_df = {
        'prompt': [],
        'interpretation': [],
        'layer': [],
        'token': [],
        'token_decoded': [],
    }

    prompt_len = original_prompt_inputs['input_ids'].shape[-1]
    outputs = model_forward_interpret(model,
                **original_prompt_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=True,
            )
    
    all_insert_infos = []
    for retrieve_layer, retrieve_token in tokens_to_interpret:
            insert_info = {}
            insert_info['replacing_mode'] = 'normalized'
            insert_info['overlay_strength'] = 1
            insert_info['retrieve_layer'] = retrieve_layer
            insert_info['retrieve_token'] = retrieve_token
            for layer_idx, layer in enumerate(model.model.layers):
                if layer_idx == k:
                    insert_locations = interpretation_prompt.insert_locations
                    insert_info[layer_idx] = (insert_locations, outputs['hidden_states'][retrieve_layer][0][retrieve_token].repeat(1,len(insert_locations), 1))
            all_insert_infos.append(insert_info)

    for batch_start_idx in tqdm(range(0,len(all_insert_infos),bs)):
        with torch.no_grad():
            batch_insert_infos = all_insert_infos[batch_start_idx:min(batch_start_idx+bs, len(all_insert_infos))]

            repeat_prompt_n_tokens = interpretation_prompt_model_inputs['input_ids'].shape[-1]
            
            batched_interpretation_prompt_model_inputs = tokenizer([interpretation_prompt.interpretation_prompt] * len(batch_insert_infos), return_tensors="pt")
            output = generate_interpret(**batched_interpretation_prompt_model_inputs, model=model, max_new_tokens=max_new_tokens, insert_info=batch_insert_infos, pad_token_id=tokenizer.eos_token_id, output_attentions = False)
            
            cropped_interpretation_tokens = output[:,repeat_prompt_n_tokens:]
            cropped_interpretation = tokenizer.batch_decode(cropped_interpretation_tokens, skip_special_tokens=True)

            for i in range(len(batch_insert_infos)):
                interpretation_df['prompt'].append(original_prompt)
                interpretation_df['interpretation'].append(cropped_interpretation[i])
                interpretation_df['layer'].append(batch_insert_infos[i]['retrieve_layer'])
                interpretation_df['token'].append(batch_insert_infos[i]['retrieve_token'])
                interpretation_df['token_decoded'].append(tokenizer.decode(original_prompt_inputs.input_ids[0, batch_insert_infos[i]['retrieve_token']]))
    return interpretation_df

    
    
