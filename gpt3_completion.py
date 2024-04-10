#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: gpt3_completion.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Wednesday April 10 2024 03:37:00
@desc: Exploring llm capabilities.
@run: python3 gpt3_completion.py
"""

import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM


class TextGenerator:
    def __init__(self, model_name: str):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
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
    # model_name = "EleutherAI/gpt-neo-125m" # 125M parameters # <If I have seen further than others, it is by> no means certain that the

    # Large model
    # model_name = "EleutherAI/gpt-neo-1.3B" # 1315M parameters # <If I have seen further than others, it is by> standing on the shoulders of giants.

    # XL model
    model_name = "EleutherAI/gpt-neo-2.7B" # 2651M parameters # <If I have seen further than others, it is by> standing on the shoulders of giants. —Isaac Newton The first time I saw the movie, I was in the middle of a long, hot summer.

    text_generator = TextGenerator(model_name)
    text_generator.print_model_info(model_name)

    prompt = "If I have seen further than others, it is by"
    generated_text = text_generator.generate_text(prompt)
    print(generated_text)


if __name__ == "__main__":
    main()