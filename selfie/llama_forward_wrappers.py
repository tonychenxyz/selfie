# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig


if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"
LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# @add_start_docstrings(
#     "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
#     LLAMA_START_DOCSTRING,
# )

@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def model_forward_interpret(
    model = None,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    insert_info = None,
    output_pre_mlp_states = False,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    # uses output attention, hidden states and return dict specified
    # when calling function or the ones from the model config
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    # print('input_ids', input_ids.shape)

    outputs, all_original_hidden_states = model_model_forward_interpret(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        insert_info = insert_info,
    )

    hidden_states = outputs[0]
    if model.config.pretraining_tp > 1:
        lm_head_slices = model.lm_head.weight.split(model.vocab_size // model.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(model.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = model.lm_head(hidden_states)
    logits = logits.float()
    # print('logits', logits.shape)
    # print('outputs', hidden_states.shape)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    
    if output_pre_mlp_states:
        if output_attentions:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                # other={'all_original_hidden_states': all_original_hidden_states}
            ), all_original_hidden_states
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), all_original_hidden_states
    if output_attentions:
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # other={'all_original_hidden_states': all_original_hidden_states}
        )
    else:
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def model_model_forward_interpret(
    model = None,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    insert_info = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # uses output attention, hidden states, use_cache and return_dict specified
    # when calling function or the ones from the model config
    output_attentions = output_attentions if output_attentions is not None else model.model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.model.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else model.model.config.use_cache

    return_dict = return_dict if return_dict is not None else model.model.config.use_return_dict
    # print('all the way from the top', input_ids.shape)
    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length # Initialize sequence length to current sequence length
    past_key_values_length = 0 # Initialize past sequence length to 0

    if past_key_values is not None:
        # If past_key_values (cached states for autoregressive decoding) is provided
        # (caching of previously calculated values for efficiency in transformers):
        # Extract length from the shape of the first entry in list
        past_key_values_length = past_key_values[0][0].shape[2] # [layers, heads, seq_length, dim]
        # Update sequence length to include past key values
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        # If `position_ids` are not provided, compute them based on the device and sequence length
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        # Create a batch dimension for `position_ids` and ensure correct shape
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        # Ensure provided position_ids are in the correct shape and data type
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        # generate embeddings from input_ids using the model's embedding layer
        inputs_embeds = model.model.embed_tokens(input_ids)
    # embed positions
    # If no attention mask is provided, create a default mask
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
        padding_mask = None # No padding mask if all tokens are valid
    else:
        # Check if the attention mask contains any 0s (indicating padding tokens)
        if 0 in attention_mask:
            padding_mask = attention_mask # Keep track of padding
        else:
            padding_mask = None # No padding mask if all tokens are valid
    # Prepare the decoders attention mask, taking past key values into account

    attention_mask = model.model._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds # Initialize hidden states with input embeddings
    original_hidden_states = (inputs_embeds, inputs_embeds) # Store original embeddings for further use

    # print("inputs_embeds", inputs_embeds.shape)

    if model.model.gradient_checkpointing and model.model.training:
        # Gradient checkpointing trades memory for computation during training
        # If gradient checkpointing is enabled, and the model is in training mode deactivate caching
        # (requires recomputation of activations during backward pass -> conflicts with caching)
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    # Initialize containers for optional outputs based on user configuration
    all_hidden_states = () if output_hidden_states else None
    all_original_hidden_states = ()
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # Loop over each decoder layer in the model
    for idx, decoder_layer in enumerate(model.model.layers):
        # Check if overlay / modification (insert_info) is to be applied and the sequence length > 1
        if hidden_states.shape[1] > 1 and insert_info != None:
            for batch_item_idx in range(len(insert_info)): # Loop through each batch item
                 # Check if  current layer idx has insert_info for this batch item
                if idx in insert_info[batch_item_idx].keys():
                    if insert_info[batch_item_idx]['replacing_mode'] == 'addition':
                        # Addition mode: overlay information is added to hidden states
                        hidden_states[batch_item_idx:batch_item_idx+1, insert_info[batch_item_idx][idx][0], :] += insert_info[batch_item_idx]['overlay_strength'] * insert_info[batch_item_idx][idx][1].to(hidden_states.device)
                    elif insert_info[batch_item_idx]['replacing_mode'] == 'normalized':
                        # Normalized mode: overlay information is combined with hidden states based on overlay_strength
                        hidden_states[batch_item_idx:batch_item_idx+1, insert_info[batch_item_idx][idx][0], :] = insert_info[batch_item_idx]['overlay_strength'] * insert_info[batch_item_idx][idx][1].to(hidden_states.device) + (1-insert_info[batch_item_idx]['overlay_strength']) * hidden_states[batch_item_idx:batch_item_idx+1, insert_info[batch_item_idx][idx][0], :]
        
        # Save hidden states (if requested)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            all_original_hidden_states += (original_hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if model.model.gradient_checkpointing and model.model.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
            )
        else:
            
            layer_outputs = decoder_layer_forward_interpret(
                hidden_states,
                decoder_layer = decoder_layer,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

        layer_outputs, original_hidden_states = layer_outputs
        hidden_states = layer_outputs[0]
        # print('model.model hiddenstates', hidden_states.shape)

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = model.model.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
        all_original_hidden_states += (original_hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    ), all_original_hidden_states


def decoder_layer_forward_interpret(
    hidden_states: torch.Tensor,
    decoder_layer=None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """

    residual = hidden_states

    hidden_states = decoder_layer.input_layernorm(hidden_states)

    # Self Attention
    # print(type(hidden_states), hidden_states.shape)
    hidden_states, self_attn_weights, present_key_value = decoder_layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )

    hidden_states = hidden_states.to(residual.device)
    # print(hidden_states.device)
    # print(residual.device)

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    pre_mlp = hidden_states
    hidden_states = decoder_layer.mlp(hidden_states)

    hidden_states = hidden_states.to(residual.device)
    # print(hidden_states.device)
    # print(residual.device)
    original_hidden_states = (pre_mlp, residual)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs, original_hidden_states