# `SelfIE`: Self-Interpretation of Large Language Model Embeddings

![SelfIE](assets/website-safety-details-fixed.svg)






This repository contains the code and data for the paper "`SelfIE`: Self-Interpretation of Large Language Model Embeddings" by [Haozhe Chen](https://tonychen.xyz/), [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/), and [Chengzhi Mao](http://www.cs.columbia.edu/~mcz/).

## Abstract
The expanding impacts of Large Language Models (LLMs) increasingly require the answer to: How do LLMs obtain their answers? The ability to explain and control an LLM's reasoning process is key for reliability, transparency, and future model developments. We propose  `SelfIE` (Self-Interpretation of Embeddings) that enables LLMs to interpret their own embeddings in natural language by leveraging their ability to respond inquiry about a given passage. Capable of interpreting open-world concepts in the hidden embeddings, `SelfIE` reveals LLM internal reasoning in cases such as making ethical decisions, internalizing prompt injection, and recalling harmful knowledge. `SelfIE`'s text descriptions on hidden embeddings also open up new avenues to control LLM reasoning. We propose Supervised Control, which allows editing open-ended concepts while only requiring gradient computation of individual layer. We extend RLHF to hidden embeddings and propose Reinforcement Control that erases harmful knowledge in LLM without supervision targets. 

## Updates
This repository is under active development. Please check back for updates. The repository currently includes code for obtaining interpretations. Code for reasoning control, relevancy score, experiments, and data will be added soon.

## Installation

To install `selfie` from the github repository main branch, run:

```bash
git clone https://github.com/tonychenxyz/selfie.git
cd selfie
pip install -e .
```

The code has been tested with `transformers==4.34.0`.

## Quickstart
Load model with huggingface. Currently the library supports all LLaMA models.
```python
from transformers import AutoTokenizer,AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
```
Create interpretation prompt from a tuple. Placeholders are denoted with `0`.
```python
from selfie.interpret import InterpretationPrompt
interpretation_prompt = InterpretationPrompt(tokenizer, ("[INST]", 0, 0, 0, 0, 0, "[/INST] Sure, I will summarize the message:"))
```
Specify original input prompt with `original_prompt` and layer and token idx to interpret in `tokens_to_interpret`. Get interpretation as a dictionary with `interpret`
```python
from selfie.interpret import interpret

original_prompt = "[INST] What's highest mountain in the world? [/INST]"
tokens_to_interpret = [(10,5), (10,6)]
bs = 2
max_new_tokens = 10
k = 1

interpretation_df = interpret(original_prompt=original_prompt, tokens_to_interpret=tokens_to_interpret, model=model, interpretation_prompt=interpretation_prompt, bs=bs, max_new_tokens=max_new_tokens, k=k, tokenizer=tokenizer)
```
See full example code in `demo.ipynb`.

## Citation
If you find this repository helpful, please consider citing our paper:
