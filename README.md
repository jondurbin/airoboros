# airoboros: using large language models to fine-tune large language models

This is my take on implementing the [Self-Instruct paper](https://arxiv.org/abs/2212.10560).  The approach is quite heavily modified, and uses the human generated seeds provided by [Databricks Dolly Project](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

This updated implementation supports either the /v1/completions endpoint or /v1/chat/completions, which is particularly useful in that it supports gpt-4 and gpt-3.5-turbo (which is 1/10 the cost of text-davinci-003).


## Key differences

* Sample instructions in prompts by default use the human-generated seeds from Dolly.
* Machine-generated instructions are not sampled for prompt examples, to avoid degredation.
* Support for either /v1/completions or /v1/chat/completions APIs (which allows gpt-3.5-turbo instead of text-davinci-003, as well as gpt-4 if you have access).
* In memory vector db (Chroma) for similarity comparison, which is much faster than calculating rouge score for each generated instruction.
* (Seemingly) better prompt, which includes injection of random topics to relate the instructions to, which creates much more diverse prompts.
* Multi-threaded producers/consumer implementation for significantly faster runtimes (generally 150+ unique prompts per minute, more initially since there are fewer duplicates, decreasing over time).
* Tries to ensure the context, if provided, is relevant to the topic and contains all the information that would be necessary to respond to the instruction, and nost just a link to article/etc.
* Generally speaking, this implementation tries to reduce some of the [noise](https://github.com/tloen/alpaca-lora/issues/65)


## Initial datasets

Be sure to look over the openai terms of service before using the datasets, specifically the section about using the outputs to generate competing models...

### 100k synthetic instructions, gpt-3.5-turbo

[instructions.jsonl](https://storage.googleapis.com/airoboros-dump/gpt-3.5-turbo-100k/instructions.jsonl)
[topics.txt](https://storage.googleapis.com/airoboros-dump/gpt-3.5-turbo-100k/topics.txt)


## Generating instructions

See available options via:
```
airoboros generate-instructions --help
```

Example output:
```
usage: self_instruct.py [-h] [--model MODEL] [--organization-id ORGANIZATION_ID] [--openai-api-key OPENAI_API_KEY] [--instruction-count INSTRUCTION_COUNT] [--seed-tasks-path SEED_TASKS_PATH] [--output-path OUTPUT_PATH] [--overwrite] [--append] [--prompt PROMPT] [--skip-instruction-re SKIP_INSTRUCTION_RE] [--code-gen-re CODE_GEN_RE]
                        [--samples-per-request SAMPLES_PER_REQUEST] [--min-instruction-length MIN_INSTRUCTION_LENGTH] [--max-instruction-length MAX_INSTRUCTION_LENGTH] [--temperature TEMPERATURE] [--top-p TOP_P] [--frequency-penalty FREQUENCY_PENALTY] [--presence-penalty PRESENCE_PENALTY] [--max-usage-tokens MAX_USAGE_TOKENS]
                        [--prompt-generation-concurrency PROMPT_GENERATION_CONCURRENCY] [--min-docsearch-score MIN_DOCSEARCH_SCORE]

options:
  -h, --help            show this help message and exit
  --model MODEL         OpenAI model/engine to use for prompt generation, which can be either part of the /v1/completions or /v1/chat/completions endpoints
  --organization-id ORGANIZATION_ID
                        organization ID to include in the request to OpenAI, defaults to organization ID tied to the API key
  --openai-api-key OPENAI_API_KEY
                        OpenAI API key to use, defaults to the OPENAI_API_KEY environment variable
  --instruction-count INSTRUCTION_COUNT
                        number of instructions to generate, not including the seed instructions
  --seed-tasks-path SEED_TASKS_PATH
                        path to an input seed instructions JSONL file
  --output-path OUTPUT_PATH
                        path to store all generated instructions in
  --overwrite           overwrite output path if it exists
  --append              append to output path if it exists
  --prompt PROMPT       prompt prefix to use for generating tasks
  --skip-instruction-re SKIP_INSTRUCTION_RE
                        regular expression used to filter low-quality/unusable instructions
  --code-gen-re CODE_GEN_RE
                        regular expression used to filter coding/programming tasks
  --samples-per-request SAMPLES_PER_REQUEST
                        number of random sample instructions to include in prompts
  --min-instruction-length MIN_INSTRUCTION_LENGTH
                        minimum instruction length
  --max-instruction-length MAX_INSTRUCTION_LENGTH
                        maximum instruction length
  --temperature TEMPERATURE
                        temperature parameter to use in OpenAI requests
  --top-p TOP_P         top-p parameter to use in OpenAI requests
  --frequency-penalty FREQUENCY_PENALTY
                        frequency penalty to use in OpenAI requests
  --presence-penalty PRESENCE_PENALTY
                        presence penalty to use in OpenAI requests
  --max-usage-tokens MAX_USAGE_TOKENS
                        Maximum token usage, calculated as sum of total_tokens from responses
  --prompt-generation-concurrency PROMPT_GENERATION_CONCURRENCY
                        Number of concurrent prompt generation threads/requests to use
  --min-docsearch-score MIN_DOCSEARCH_SCORE
```

## Coming soon

Scripts for fine-tuning various models using the self-instruct (and human-generated) prompts.
