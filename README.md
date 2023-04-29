# Ouroboros: Aligning large language models with self (openai) generated instructions.

This repository is my own take on implementing the [Self-Instruct paper](https://arxiv.org/abs/2212.10560).  The approach is quite heavily modified, and uses the seeds provided by [Databricks Dolly Project](https://huggingface.co/datasets/databricks/databricks-dolly-15k)


## Generating instructions

See available options via:
```
python -m ouroboros.self_instruct --help
```

Example output:
```
usage: self_instruct.py [-h] [--model MODEL]
                        [--organization-id ORGANIZATION_ID]
                        [--openai-api-key OPENAI_API_KEY]
                        [--instruction-count INSTRUCTION_COUNT]
                        [--seed-tasks-path SEED_TASKS_PATH]
                        [--output-path OUTPUT_PATH] [--overwrite]
                        [--default-prompt-prefix DEFAULT_PROMPT_PREFIX]
                        [--classification-prompt-prefix CLASSIFICATION_PROMPT_PREFIX]
                        [--contextual-prompt-prefix CONTEXTUAL_PROMPT_PREFIX]
                        [--skip-instruction-re SKIP_INSTRUCTION_RE]
                        [--code-gen-re CODE_GEN_RE]
                        [--min-instruction-length MIN_INSTRUCTION_LENGTH]
                        [--max-instruction-length MAX_INSTRUCTION_LENGTH]
                        [--max-tokens MAX_TOKENS] [--temperature TEMPERATURE]
                        [--top-p TOP_P]
                        [--frequency-penalty FREQUENCY_PENALTY]
                        [--presence-penalty PRESENCE_PENALTY]
                        [--max-usage-tokens MAX_USAGE_TOKENS]

options:
  -h, --help            show this help message and exit
  --model MODEL         OpenAI model/engine to use for prompt generation,
                        which can be either part of the /v1/completions or
                        /v1/chat/completions endpoints
  --organization-id ORGANIZATION_ID
                        organization ID to include in the request to OpenAI,
                        defaults to organization ID tied to the API key
  --openai-api-key OPENAI_API_KEY
                        OpenAI API key to use, defaults to the OPENAI_API_KEY
                        environment variable
  --instruction-count INSTRUCTION_COUNT
                        number of instructions to generate, not including the
                        seed instructions
  --seed-tasks-path SEED_TASKS_PATH
                        path to an input seed instructions JSONL file
  --output-path OUTPUT_PATH
                        path to store all generated instructions in
  --overwrite           overwrite output path if it exists
  --default-prompt-prefix DEFAULT_PROMPT_PREFIX
                        prompt prefix to use for generating open, non-
                        classification tasks
  --classification-prompt-prefix CLASSIFICATION_PROMPT_PREFIX
                        prompt prefix to use for generating classification
                        tasks
  --contextual-prompt-prefix CONTEXTUAL_PROMPT_PREFIX
                        prompt prefix to use for generating tasks with
                        context, e.g. closed Q&A
  --skip-instruction-re SKIP_INSTRUCTION_RE
                        regular expression used to filter low-quality/unusable
                        instructions
  --code-gen-re CODE_GEN_RE
                        regular expression used to filter coding/programming
                        tasks
  --min-instruction-length MIN_INSTRUCTION_LENGTH
                        minimum instruction length
  --max-instruction-length MAX_INSTRUCTION_LENGTH
                        maximum instruction length
  --max-tokens MAX_TOKENS
                        maximum number of tokens in an instruction
  --temperature TEMPERATURE
                        temperature parameter to use in OpenAI requests
  --top-p TOP_P         top-p parameter to use in OpenAI requests
  --frequency-penalty FREQUENCY_PENALTY
                        frequency penalty to use in OpenAI requests
  --presence-penalty PRESENCE_PENALTY
                        presence penalty to use in OpenAI requests
  --max-usage-tokens MAX_USAGE_TOKENS
                        Maximum token usage, calculated as sum of total_tokens
                        from responses
```
