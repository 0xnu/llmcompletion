#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: gpt2_completion.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Wednesday April 10 2024 03:15:00
@desc: Exploring llm capabilities.
@run: python3 gpt2_completion.py
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class TextGenerator:
    def __init__(self, model_name: str):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def generate_text(
        self,
        prompt: str,
        num_sentences: int = 2,
        max_tokens: int = 250,
        temperature: float | None = None,
    ) -> str:
        token_count, sentence_count = 0, 0
        tokens = self.tokenizer.encode(prompt)
        prompt_length = len(tokens)

        while True:
            outputs = self.model(torch.tensor([tokens])).logits
            if temperature is None:
                next_token = torch.argmax(outputs[0, -1])
            else:
                dist = torch.distributions.Categorical(logits=outputs[0, -1] / temperature)
                next_token = dist.sample((1,)).item()

            tokens.append(next_token)
            token_count += 1
            if self.tokenizer.decode([next_token]) == ".":
                sentence_count += 1
            if sentence_count == num_sentences or token_count == max_tokens:
                return (
                    "<"
                    + self.tokenizer.decode(tokens[:prompt_length])
                    + ">"
                    + self.tokenizer.decode(tokens[prompt_length:])
                )

    def print_model_info(self, model_name: str) -> None:
        num_parameters = int(sum(p.numel() for p in self.model.parameters()) / (1e6))
        print(f"Using {model_name} ({num_parameters}M parameters)")


def main() -> None:
    # Base model
    # model_name = "gpt2" # 124M parameters # <If I have seen further than others, it is by> no means certain that I have seen the same thing.

    # Large model
    # model_name = "gpt2-large" # 774M parameters # <If I have seen further than others, it is by> standing on the shoulders of giants.

    # XL model
    model_name = "gpt2-xl" # 1557M parameters # <If I have seen further than others, it is by> standing on the shoulders of giants.

    text_generator = TextGenerator(model_name)
    text_generator.print_model_info(model_name)

    prompt = "If I have seen further than others, it is by"
    generated_text = text_generator.generate_text(prompt)
    print(generated_text)


if __name__ == "__main__":
    main()