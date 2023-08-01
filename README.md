# airoboros: using large language models to fine-tune large language models

This is my take on implementing the [Self-Instruct paper](https://arxiv.org/abs/2212.10560).  The approach is quite heavily modified, and does not use any human-generated seeds.

This updated implementation supports either the /v1/completions endpoint or /v1/chat/completions, which is particularly useful in that it supports gpt-4 and gpt-3.5-turbo (which is 1/10 the cost of text-davinci-003).


## Key differences

* support for either /v1/completions or /v1/chat/completions APIs (which allows gpt-3.5-turbo instead of text-davinci-003, as well as gpt-4 if you have access)
* support for custom topics list, custom topic generation prompt, or completely random topics
* in-memory vector db (Chroma) for similarity comparison, which is much faster than calculating rouge score for each generated instruction
* (seemingly) better prompts, which includes injection of random topics to relate the instructions to, which creates much more diverse synthetic instructions
* asyncio producers with configurable batch size
* several "instructors", each targetting specific use-cases, such as Orca style reasoning/math, role playing, etc.
* tries to ensure the context, if provided, is relevant to the topic and contains all the information that would be necessary to respond to the instruction, and nost just a link to article/etc.
* generally speaking, this implementation tries to reduce some of the [noise](https://github.com/tloen/alpaca-lora/issues/65)


## Generating instructions

### NEW - 2023-07-18

To better accomodate the plethora of options, the configuration has been moved to a YAML config file.

Please create a copy of `example-config.yaml` and configure as desired.

Once you have the desired configuration, run:

```
airoboros generate-instructions --config-path /path/to/config.yaml
```

## Generating topics

### NEW - 2023-07-18

Again, this is now all YAML configuration based!  Please create a customized version of the YAML config file, then run:

```
airoboros generate-topics --config-path /path/to/config.yaml
```

You can override the `topic_prompt` string in the configuration to use a different topic generation prompt.


## Support the work

https://bmc.link/jondurbin

ETH
0xce914eAFC2fe52FdceE59565Dd92c06f776fcb11

BTC
bc1qdwuth4vlg8x37ggntlxu5cjfwgmdy5zaa7pswf

## Models (research use only):

### gpt-4 versions

#### llama-2 base model

Latest version (2.0 / m2.0 datasets)
* [airoboros-l2-7b-gpt4-2.0](https://huggingface.co/jondurbin/airoboros-l2-7b-gpt4-2.0)
* [airoboros-l2-7b-gpt4-m2.0](https://huggingface.co/jondurbin/airoboros-l2-7b-gpt4-m2.0)
* [airoboros-l2-13b-gpt4-2.0](https://huggingface.co/jondurbin/airoboros-l2-13b-gpt4-2.0)
* [airoboros-l2-13b-gpt4-m2.0](https://huggingface.co/jondurbin/airoboros-l2-13b-gpt4-m2.0)

Previous generation (1.4.1 dataset)
* [airoboros-l2-70b-gpt4-1.4.1](https://huggingface.co/jondurbin/airoboros-l2-70b-gpt4-1.4.1)
* [airoboros-l2-13b-gpt4-1.4.1](https://huggingface.co/jondurbin/airoboros-l2-13b-gpt4-1.4.1)
* [airoboros-l2-7b-gpt4-1.4.1](https://huggingface.co/jondurbin/airoboros-l2-7b-gpt4-1.4.1)

#### original llama base model

Latest version (2.0 / m2.0 datasets)
* [airoboros-33b-gpt4-2.0](https://huggingface.co/jondurbin/airoboros-33b-gpt4-2.0)
* [airoboros-33b-gpt4-m2.0](https://huggingface.co/jondurbin/airoboros-33b-gpt4-m2.0)

Previous generation (1.4.1 dataset)
* [airoboros-65b-gpt4-1.4](https://huggingface.co/jondurbin/airoboros-65b-gpt4-1.4)
* [airoboros-33b-gpt4-1.4](https://huggingface.co/jondurbin/airoboros-33b-gpt4-1.4)
* [airoboros-13b-gpt4-1.4](https://huggingface.co/jondurbin/airoboros-13b-gpt4-1.4)
* [airoboros-7b-gpt4-1.4](https://huggingface.co/jondurbin/airoboros-7b-gpt4-1.4)
* *older versions on HF as well*

#### mpt-30b base model
* [airoboros-mpt-30bb-gpt4-1.4](https://huggingface.co/jondurbin/airoboros-mpt-30b-gpt4-1p4-five-epochs)

### gpt-3.5-turbo versions
* [airoboros-gpt-3.5-turbo-100k-7b](https://huggingface.co/jondurbin/airoboros-gpt-3.5-turbo-100k-7b)
* [airoboros-13b](https://huggingface.co/jondurbin/airoboros-13b)
* [airoboros-7b](https://huggingface.co/jondurbin/airoboros-7b)

## Datasets (subject to OpenAI license):

* [airoboros-gpt-3.5-turbo](https://huggingface.co/datasets/jondurbin/airoboros-uncensored)
* [airoboros-gpt4](https://huggingface.co/datasets/jondurbin/airoboros-gpt4)
* [airoboros-gpt4-1.1](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.1)
* [airoboros-gpt4-1.2](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.2)
* [airoboros-gpt4-1.3](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.3)
* [airoboros-gpt4-1.4](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.4)
* [airoboros-gpt4-2.0 (June only GPT4)](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-2.0)
* [airoboros-gpt4-m2.0 (recommended)](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-m2.0)

## Coming soon

Scripts for fine-tuning various models using the self-instruct (and human-generated) prompts.
