# airoboros: using large language models to fine-tune large language models

This is my take on implementing the [Self-Instruct paper](https://arxiv.org/abs/2212.10560).  The approach is quite heavily modified, and does not use any human-generated seeds.

This updated implementation supports either the /v1/completions endpoint or /v1/chat/completions, which is particularly useful in that it supports gpt-4 and gpt-3.5-turbo (which is 1/10 the cost of text-davinci-003).

Huge thank you to the folks over at [a16z](https://a16z.com/) for sponsoring the costs associated with building models and associated tools!

## Install

via pip:
```
pip install --no-build-isolation airoboros
```

from source (keeping the source):
```
git clone https://github.com/jondurbin/airoboros
pip install -e --no-build-isolation ./airoboros
```

## Key differences from self-instruct/alpaca

* support for either /v1/completions or /v1/chat/completions APIs (which allows gpt-3.5-turbo instead of text-davinci-003, as well as gpt-4 if you have access)
* support for custom topics list, custom topic generation prompt, or completely random topics
* in-memory vector db (Chroma) for similarity comparison, which is much faster than calculating rouge score for each generated instruction
* (seemingly) better prompts, which includes injection of random topics to relate the instructions to, which creates much more diverse synthetic instructions
* asyncio producers with configurable batch size
* several "instructors", each targetting specific use-cases, such as Orca style reasoning/math, role playing, etc.
* tries to ensure the context, if provided, is relevant to the topic and contains all the information that would be necessary to respond to the instruction, and nost just a link to article/etc.
* generally speaking, this implementation tries to reduce some of the [noise](https://github.com/tloen/alpaca-lora/issues/65)

## Goal of this project

Problem and proposed solution:

- Models can only ever be as good as the data they are trained on.
- High quality data is difficult to curate manually, so ideally the process can be automated by AI/LLMs.
- Large models (gpt-4, etc.) are pricey to build/run and out of reach for individuals/small-medium business, and are subject to RLHF bias, censorship, and changes without notice.
- Smaller models (llama-2-70b, etc.) can reach somewhat comparable performance in specific tasks to much larger models when trained on high quality data.
- The airoboros tool allows building datasets that are focused on specific tasks, which can then be used to build a plethora of individual expert models.  This means we can crowdsource building experts.
- Using either a classifier model, or simply calculating vector embeddings for each item in the dataset and using faiss index/cosine similarity/etc. search, incoming requests can be routed to a particular expert (e.g. dynamically loading LoRAs) to get extremely high quality responses.

Progress:

- ✅ PoC that training via self-instruction, that is, datasets generated from language models, works reasonably well.
- ✅ Iterate on the PoC to use higher quality prompts, more variety of instructions, etc.
- ✅ Split the code into separate "instructors", for specializing in any particular task (creative writing, songs, roleplay, coding, execution planning, function calling, etc.)
- [in progress]: PoC that an ensemble of LoRAs split by the category (i.e., the instructor used in airoboros) has better performance than the same param count model tuned on all data
- [in progress]: Remove the dependency on OpenAI/gpt-4 to generate the training data so all datasets can be completely free and open source.
- [future]: Automatic splitting of experts at some threshold, e.g. "coding" is split into python, js, golang, etc.
- [future]: Hosted service/site to build and/or extend datasets or models using airoboros.
- [future]: Depending on success of all of the above, potentially a hosted inference option with an exchange for private/paid LoRAs.

## LMoE

<img src="https://github.com/jondurbin/airoboros/blob/main/assets/lmoe.jpeg" alt="LMoE" width="300" class="center"/>

LMoE is the simplest architecture I can think of for a mixture of experts.  It doesn't use a switch transformer, doesn't require slicing and merging layers with additional fine-tuning, etc.  It just dynamically loads the best PEFT/LoRA adapter model based on the incoming request.

By using this method, we can theoretically crowdsource generation of dozens (or hundreds/thousands?) of very task-specific adapters and have an extremely powerful ensemble of models with very limited resources on top of a single base model (llama-2 7b/13b/70b).

### Tuning the experts

The self-instruct code contained within this project uses many different "instructors" to generate training data to accomplish specific tasks.  The output includes the instructor/category that generated the data.  We can use this to automatically segment the training data to fine-tune specific "experts".

See `scripts/segment_experts.py` for an example of how the training data can be segmented, with a sampling of each other expert in the event of misrouting.

See `scripts/tune_expert.py` for an example of creating the adapter models (with positional args for expert name, model size, etc.)

__*NOTE: this assumes use of my fork of qlora https://github.com/jondurbin/qlora*__

### Routing requests to the expert

The "best" routing mechanism would probably be to train a classifier based on the instructions for each category, with the category/expert being the label, but that prohibits dynamic loading of new experts.

Instead, this supports 3 options:

- faiss index similarity search using the training data for each expert (default)
- agent-based router using the "function" expert (query the LLM with a list of available experts and their descriptions, ask which would be best based on the user's input)
- specify the agent in the JSON request

### Running the API server

First, download the base llama-2 model for whichever model size you want, e.g.: [llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)

Next, download the LMoE package that corresponds to that base model, e.g.: [airoboros-lmoe-7b-2.1](https://huggingface.co/jondurbin/airoboros-lmoe-7b-2.1)

*NOTE: 13b also available, 70b in progress*

Here's an example command to start the server:

```
python -m airoboros.lmoe.api \
  --base-model ./llama-2-7b-hf \
  --lmoe ./airoboros-lmoe-7b-2.1 \
  --router-max-samples 1000 \
  --router-k 25 \
  --port 8000 \
  --host 127.0.0.1
```
*to use the agent-based router, add `--agent-router` to the arguments*

This uses flash attention via bettertransformers (in optimum).  You may need to install torch nightly if you see an error like 'no kernel available', e.g.:

```
pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

Once started, you can infer using the same API scheme you'd query OpenAI API with, e.g.:

```
curl -H 'content-type: application/json' http://127.0.0.1:8000/v1/chat/completions -d '
{
  "model": "llama-2-7b-hf",
  "temperature": 0.7,
  "max_tokens": 2048,
  "messages": [
    {
      "role": "system",
      "content": "A chat."
    },
    {
      "role": "user",
      "content": "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
    }
  ]
}'
```

I've also added an vllm-based server, but the results aren't quite as good (not sure why yet).  To use it, make sure you install `vllm` and `fschat`, or `pip install airoboros[vllm]`

```
python -m airoboros.lmoe.vllm \
  --model ./llama-2-7b-hf \
  --lmoe-path ../airoboros-lmoe-7b-2.1 \
  --router-max-samples 100 \
  --router-k 25 \
  --port 8000 \
  --host 127.0.0.1
```

## Generating instructions

### NEW - 2023-07-18

To better accommodate the plethora of options, the configuration has been moved to a YAML config file.

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

2.1 dataset
* [airoboros-l2-7b-2.1](https://huggingface.co/jondurbin/airoboros-l2-7b-2.1)
* [airoboros-l2-13b-2.1](https://huggingface.co/jondurbin/airoboros-l2-13b-2.1)
* [airoboros-l2-70b-2.1](https://huggingface.co/jondurbin/airoboros-l2-70b-2.1)
* [airoboros-c34b-2.1](https://huggingface.co/jondurbin/airoboros-c34b-2.1)

2.0/m2.0
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

## Datasets

* [airoboros-gpt-3.5-turbo](https://huggingface.co/datasets/jondurbin/airoboros-uncensored)
* [airoboros-gpt4](https://huggingface.co/datasets/jondurbin/airoboros-gpt4)
* [airoboros-gpt4-1.1](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.1)
* [airoboros-gpt4-1.2](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.2)
* [airoboros-gpt4-1.3](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.3)
* [airoboros-gpt4-1.4](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.4)
* [airoboros-gpt4-2.0 (June only GPT4)](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-2.0)
* [airoboros-gpt4-m2.0](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-m2.0)
* [airoboros-2.1 (recommended)](https://huggingface.co/datasets/jondurbin/airoboros-2.1)
