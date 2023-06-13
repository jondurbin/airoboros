# airoboros: using large language models to fine-tune large language models

This is my take on implementing the [Self-Instruct paper](https://arxiv.org/abs/2212.10560).  The approach is quite heavily modified, and does not use any human-generated seeds.

This updated implementation supports either the /v1/completions endpoint or /v1/chat/completions, which is particularly useful in that it supports gpt-4 and gpt-3.5-turbo (which is 1/10 the cost of text-davinci-003).


## Key differences

* support for either /v1/completions or /v1/chat/completions APIs (which allows gpt-3.5-turbo instead of text-davinci-003, as well as gpt-4 if you have access)
* support for custom topics list, custom topic generation prompt, or completely random topics
* in-memory vector db (Chroma) for similarity comparison, which is much faster than calculating rouge score for each generated instruction
* (seemingly) better prompts, which includes injection of random topics to relate the instructions to, which creates much more diverse synthetic instructions
* multi-threaded producer/consumer implementation for significantly faster runtimes (generally 150+ unique prompts per minute, more initially since there are fewer duplicates, decreasing over time).
* tries to ensure the context, if provided, is relevant to the topic and contains all the information that would be necessary to respond to the instruction, and nost just a link to article/etc.
* generally speaking, this implementation tries to reduce some of the [noise](https://github.com/tloen/alpaca-lora/issues/65)


## Generating instructions

See available options via:
```
airoboros generate-instructions --help
```

Help as of 2023-05-12:
```
usage: self_instruct.py [-h] [--model MODEL] [--organization-id ORGANIZATION_ID] [--openai-api-key OPENAI_API_KEY] [--instruction-count INSTRUCTION_COUNT]
                        [--batch-size BATCH_SIZE] [--output-path OUTPUT_PATH] [--topics-path TOPICS_PATH] [--overwrite] [--append] [--uncensored]
                        [--bot-name BOT_NAME] [--prompt PROMPT] [--contextual-prompt CONTEXTUAL_PROMPT] [--topic-generation-prompt TOPIC_GENERATION_PROMPT]
                        [--topic-request-count TOPIC_REQUEST_COUNT] [--contextual-prompt-ratio CONTEXTUAL_PROMPT_RATIO] [--skip-instruction-re SKIP_INSTRUCTION_RE]
                        [--temperature TEMPERATURE] [--prompt-generation-temperature PROMPT_GENERATION_TEMPERATURE] [--top-p TOP_P]
                        [--frequency-penalty FREQUENCY_PENALTY] [--presence-penalty PRESENCE_PENALTY] [--max-usage-tokens MAX_USAGE_TOKENS]
                        [--concurrency CONCURRENCY] [--min-docsearch-score MIN_DOCSEARCH_SCORE]

options:
  -h, --help            show this help message and exit
  --model MODEL         OpenAI model/engine to use for prompt generation, which can be either part of the /v1/completions or /v1/chat/completions endpoints
  --organization-id ORGANIZATION_ID
                        organization ID to include in the request to OpenAI, defaults to organization ID tied to the API key
  --openai-api-key OPENAI_API_KEY
                        OpenAI API key to use, defaults to the OPENAI_API_KEY environment variable
  --instruction-count INSTRUCTION_COUNT
                        number of instructions to generate, not including the seed instructions
  --batch-size BATCH_SIZE
                        number of candidate instructions to (attempt to) generate per request
  --output-path OUTPUT_PATH
                        path to store all generated instructions in
  --topics-path TOPICS_PATH
                        path to a newline separated list of topics
  --overwrite           overwrite output path if it exists
  --append              append to output path if it exists
  --uncensored          try to produce uncensored responses, via role-play prompt
  --bot-name BOT_NAME   name of the bot, when using uncensored mode
  --prompt PROMPT       prompt prefix to use for generating non-contextual instructions
  --contextual-prompt CONTEXTUAL_PROMPT
                        prompt to use for generating contextual prompts
  --topic-generation-prompt TOPIC_GENERATION_PROMPT
                        prompt to use in generating random topics
  --topic-request-count TOPIC_REQUEST_COUNT
                        number of requests to perform in random topic generation
  --contextual-prompt-ratio CONTEXTUAL_PROMPT_RATIO
                        ratio of prompts that should be contextual, e.g. summarization of an article
  --skip-instruction-re SKIP_INSTRUCTION_RE
                        regular expression used to filter low-quality/unusable instructions
  --temperature TEMPERATURE
                        temperature parameter to use in OpenAI requests to generate responses
  --prompt-generation-temperature PROMPT_GENERATION_TEMPERATURE
                        temperature parameter to use in OpenAI requests when generating synthetic instructions
  --top-p TOP_P         top-p parameter to use in OpenAI requests
  --frequency-penalty FREQUENCY_PENALTY
                        frequency penalty to use in OpenAI requests
  --presence-penalty PRESENCE_PENALTY
                        presence penalty to use in OpenAI requests
  --max-usage-tokens MAX_USAGE_TOKENS
                        Maximum token usage, calculated as sum of total_tokens from responses
  --concurrency CONCURRENCY
                        Number of concurrent threads/requests to use
  --min-docsearch-score MIN_DOCSEARCH_SCORE
```

### Using custom topics:

If you want to use a specific set of topics for prompt generation, add `--topics-path /path/to/topics.txt` to the command.  This file must be plain text, one topic per line.

### Using a custom topic generator:

If you want to use random topics, but want those topics to be somewhat related to a specific category or idea, try playing with `--topic-generation-prompt` (and probablyl `--topic-request-count`).  By default, the topic generation prompt is just random, but you can try things like:

```
... --topic-generation-prompt "Give me a list of 100 significant historical battles." --topic-request-count 10
```

Since the returned topics may include duplicates, it is not guaranteed that your topic list will contain 100 * 10 topics.


## Support the work

https://bmc.link/jondurbin

## Models (research use only):

### gpt-4 versions
* [airoboros-33b-gpt4](https://huggingface.co/jondurbin/airoboros-33b-gpt4)
* [airoboros-13b-gpt4-1.1](https://huggingface.co/jondurbin/airoboros-13b-gpt4-1.1)
* [airoboros-7b-gpt4-1.1](https://huggingface.co/jondurbin/airoboros-7b-gpt4-1.1)

### gpt-3.5-turbo versions
* [airoboros-gpt-3.5-turbo-100k-7b](https://huggingface.co/jondurbin/airoboros-gpt-3.5-turbo-100k-7b)
* [airoboros-13b](https://huggingface.co/jondurbin/airoboros-13b)
* [airoboros-7b](https://huggingface.co/jondurbin/airoboros-7b)

## Datasets (subject to OpenAI license):

* [airoboros-gpt-3.5-turbo](https://huggingface.co/datasets/jondurbin/airoboros-uncensored)
* [airoboros-gpt4](https://huggingface.co/datasets/jondurbin/airoboros-gpt4)
* [airoboros-gpt4-1.1](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.1)
* [airoboros-gpt4-1.2 (recommended)](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.2)

## Coming soon

Scripts for fine-tuning various models using the self-instruct (and human-generated) prompts.
