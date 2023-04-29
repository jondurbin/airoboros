import asyncio
import aiohttp
import argparse
import backoff
import copy
import os
import json
import random
import re
import requests
import string
from loguru import logger
from typing import List, Dict, Any
from .exceptions import (
    RateLimitError,
    TooManyRequestsError,
    TokensExhaustedError,
    ContextLengthExceededError,
    BadResponseError,
)

# Defaults and constants.
CLASSIFICATION_PROMPT_PREFIX = "Come up with a series of classification tasks, including the list of possible output labels."
DEFAULT_PROMPT_PREFIX = "Come up with a series of tasks:"
CONTEXTUAL_PROMPT_PREFIX = "Come up with a series of tasks, including a passage of text that could be used to respond to the task:"
SKIP_WORDS = ["image", "graph", "picture", "file", "map", "draw", "plot", "go to"]
SKIP_SEARCH_RE = re.compile(r"\b{'|'.join(SKIP_WORDS)}s?\b", re.I)
CONTEXT_RE = re.compile(r"[r\n\s]*(?:\d+\s*\.\s*)?(.*)[\r\n\s]passage:\s*(.*)", re.I)
CODE_GEN_RE = re.compile(
    r"^(?:write|generate|create)(?:a )?(?:\w+ )?(?:script|program|code)\W", re.I
)
DOLLY_SEED_URL = "https://raw.githubusercontent.com/databrickslabs/dolly/master/data/databricks-dolly-15k.jsonl"
OPENAI_API_BASE_URL = "https://api.openai.com"
MODEL_ENDPOINTS = {
    "completions": [
        "text-davinci-003",
        "text-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
    ],
    "chat_completions": [
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
    ],
}


class SelfInstructor:
    """Class and methods used to generate instructions, based on self-instruct paper/code."""

    CLI_ARGS = {
        "--model": {
            "type": str,
            "default": "gpt-3.5-turbo",
            "help": "OpenAI model/engine to use for prompt generation, which can be either part of the /v1/completions or /v1/chat/completions endpoints",
        },
        "--organization-id": {
            "type": str,
            "help": "organization ID to include in the request to OpenAI, defaults to organization ID tied to the API key",
        },
        "--openai-api-key": {
            "type": str,
            "help": "OpenAI API key to use, defaults to the OPENAI_API_KEY environment variable",
        },
        "--instruction-count": {
            "type": int,
            "default": 100000,
            "help": "number of instructions to generate, not including the seed instructions",
        },
        "--seed-tasks-path": {
            "type": str,
            "help": "path to an input seed instructions JSONL file",
        },
        "--output-path": {
            "type": str,
            "help": "path to store all generated instructions in",
        },
        "--overwrite": {
            "action": "store_true",
            "help": "overwrite output path if it exists",
        },
        "--default-prompt-prefix": {
            "type": str,
            "default": DEFAULT_PROMPT_PREFIX,
            "help": "prompt prefix to use for generating open, non-classification tasks",
        },
        "--classification-prompt-prefix": {
            "type": str,
            "default": CLASSIFICATION_PROMPT_PREFIX,
            "help": "prompt prefix to use for generating classification tasks",
        },
        "--contextual-prompt-prefix": {
            "type": str,
            "default": CONTEXTUAL_PROMPT_PREFIX,
            "help": "prompt prefix to use for generating tasks with context, e.g. closed Q&A",
        },
        "--skip-instruction-re": {
            "type": str,
            "default": SKIP_SEARCH_RE.pattern,
            "help": "regular expression used to filter low-quality/unusable instructions",
        },
        "--code-gen-re": {
            "type": str,
            "default": CODE_GEN_RE.pattern,
            "help": "regular expression used to filter coding/programming tasks",
        },
        "--min-instruction-length": {
            "type": int,
            "default": 3,
            "help": "minimum instruction length",
        },
        "--max-instruction-length": {
            "type": int,
            "default": 150,
            "help": "maximum instruction length",
        },
        "--max-tokens": {
            "type": int,
            "default": 1024,
            "help": "maximum number of tokens in an instruction",
        },
        "--temperature": {
            "type": float,
            "default": 0.7,
            "help": "temperature parameter to use in OpenAI requests",
        },
        "--top-p": {
            "type": float,
            "default": 0.5,
            "help": "top-p parameter to use in OpenAI requests",
        },
        "--frequency-penalty": {
            "type": int,
            "default": 0,
            "help": "frequency penalty to use in OpenAI requests",
        },
        "--presence-penalty": {
            "type": int,
            "default": 2,
            "help": "presence penalty to use in OpenAI requests",
        },
        "--max-usage-tokens": {
            "type": int,
            "help": "Maximum token usage, calculated as sum of total_tokens from responses",
        },
    }

    def __init__(
        self,
        *,
        model: str = "gpt-3.5-turbo",
        organization_id: str = None,
        openai_api_key: str = None,
        instruction_count: int = 100000,
        seed_tasks: List[Dict[str, str]] = None,
        seed_tasks_path: str = None,
        output_path: str = None,
        overwrite: bool = False,
        use_dolly_seed: bool = True,
        classification_prompt_prefix: str = CLASSIFICATION_PROMPT_PREFIX,
        default_prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
        contextual_prompt_prefix: str = CONTEXTUAL_PROMPT_PREFIX,
        skip_instruction_re: re.Pattern = SKIP_SEARCH_RE,
        code_gen_re: re.Pattern = CODE_GEN_RE,
        min_instruction_length: int = 3,
        max_instruction_length: int = 150,
        instructions_per_request: int = 7,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.5,
        frequency_penalty: int = 0,
        presence_penalty: int = 2,
        max_usage_tokens: int | None = None,
    ):
        """Constructor."""
        self.model = model
        self.organization_id = organization_id
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable or openai_api_key must be provided"
            )
        self.instruction_count = instruction_count
        if not output_path:
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "output",
                "instructions.jsonl",
            )
        self.output_path = output_path
        self.overwrite = overwrite
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.classification_prompt_prefix = classification_prompt_prefix
        self.default_prompt_prefix = default_prompt_prefix
        self.contextual_prompt_prefix = contextual_prompt_prefix
        self.skip_instruction_re = skip_instruction_re
        if isinstance(skip_instruction_re, str):
            self.skip_instruction_re = re.compile(skip_instruction_re, re.I)
        self.code_gen_re = code_gen_re
        if isinstance(code_gen_re, str):
            self.code_gen_re = re.compile(code_gen_re, re.I)
        self.min_instruction_length = min_instruction_length
        self.max_instruction_length = max_instruction_length
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.instructions_per_request = instructions_per_request
        self.max_usage_tokens = max_usage_tokens
        self._validate_model()
        self._initialize_seed_tasks(seed_tasks, seed_tasks_path, use_dolly_seed)
        self.classification_seed_tasks = []
        self.contextual_seed_tasks = []
        self.open_seed_tasks = []
        self.machine_tasks = []
        self.classification_machine_tasks = []
        self.contextual_machine_tasks = []
        self.open_machine_tasks = []
        self.used_tokens = 0
        for task in self.seed_tasks:
            if task["category"] == "classification":
                self.classification_seed_tasks.append(task)
            elif task.get("context", "").strip():
                self.contextual_seed_tasks.append(task)
            else:
                self.open_seed_tasks.append(task)
        self.classification_ratio = len(self.classification_seed_tasks) / len(
            self.seed_tasks
        )
        self.contextual_ratio = len(self.contextual_seed_tasks) / len(self.seed_tasks)

    def _initialize_seed_tasks(
        self,
        seed_tasks: List[Dict[str, str]],
        seed_tasks_path: str,
        use_dolly_seed: bool,
    ):
        """Helper method to construct the seed tasks, either as input dict, from
        user-provided seed path, or using the dolly 15k seeds (default).
        """
        if seed_tasks:
            self.seed_tasks = seed_tasks
            return
        if seed_tasks_path:
            seed_tasks = []
            with open(seed_tasks_path) as infile:
                for line in infile.readlines():
                    seed_tasks.append(json.loads(line))
        elif use_dolly_seed:
            dolly_seed_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), ".seed", "dolly_seeds.jsonl"
            )
            if not os.path.isfile(dolly_seed_path):
                result = requests.get(DOLLY_SEED_URL)
                with open(dolly_seed_path, "w") as outfile:
                    outfile.write(result.text)
            with open(dolly_seed_path, "r") as infile:
                seed_tasks = [json.loads(line) for line in infile.readlines()]
            self.seed_tasks = seed_tasks
        logger.info(f"Found {len(self.seed_tasks)} seed tasks to use...")

    def _validate_model(self):
        """Ensure the specified model is available, and configure the endpoint
        to use accordingly (chat completions or completions).
        """
        if self.model in MODEL_ENDPOINTS["completions"]:
            self._completions = True
        elif self.model in MODEL_ENDPOINTS["chat_completions"]:
            self._completions = False
        else:
            raise ValueError(f"Model is not currently supported: {self.model}")
        # Ensure the model is authorized for this key.
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        result = requests.get(f"{OPENAI_API_BASE_URL}/v1/models", headers=headers)
        if result.status_code != 200:
            raise ValueError(
                f"Invalid openai API key [{result.status_code}: {result.text}]"
            )
        available = {item["id"] for item in result.json()["data"]}
        if self.model not in available:
            raise ValueError(f"Model is not available to your API key: {self.model}")
        logger.success(f"Successfully validated model: {self.model}")

    @staticmethod
    def clean_instruction_text(instruction: str) -> str:
        """Remove extra/trailing whitespace from instruction prompts.

        :param instruction: The prompt text to clean.
        :type instruction: str

        :return: The cleaned prompt text.
        :rtype: str
        """
        return re.sub(
            r"\s+", " ", " ".join(instruction.splitlines()).strip().rstrip(":")
        )

    def generate_prompt_from_instructions(
        self, instructions: List[Dict[str, any]], instruction_type: str
    ) -> str:
        """Generate a single prompt string with multiple instructions.

        :param instructions: A list of instructions to encode.
        :type instructions: List[str]

        :param instruction_type: The type of instruction (classification, open, contextual)
        :type: instruction_type: str

        :return: The encoded prompt.
        :rtype: str
        """
        prompt = []
        if instruction_type == "classification":
            prompt.append(self.classification_prompt_prefix)
        elif instruction_type == "contextual":
            prompt.append(self.contextual_prompt_prefix)
        else:
            prompt.append(self.default_prompt_prefix)
        prompt.append("\n")
        for idx, instruction in enumerate(instructions):
            text = self.clean_instruction_text(instruction["instruction"])
            if instruction.get("context", "").strip():
                text += (
                    f"\nPassage: {self.clean_instruction_text(instruction['context'])}"
                )
            prompt.append(f"{idx+1}. {text}\n" + "=" * 8)
        prompt.append(f"{len(instructions) + 1}.")
        return "\n".join(prompt)

    def extract_instructions_from_response(
        self, response: Dict[str, Any], instruction_type: str
    ) -> List[Dict[str, Any]]:
        """Extract the list of instructions from the OpenAI response.

        :param response: The response from the OpenAI request.
        :type re: Dict[str, Any]

        :param instruction_type: The type of instruction we are expecting.
        :type instruction_type: str

        :return: List of instructions.
        :rtype: List[str]
        """
        if (
            not response
            or not response.get("choices")
            or response["choices"][0]["finish_reason"] == "length"
        ):
            return []

        text = None
        if self._completions:
            text = response["choices"][0]["text"]
        else:
            text = response["choices"][0]["message"]["content"]
        raw_instructions = text.split("=" * 8)
        instructions = []
        for raw_instruction in raw_instructions:
            context = ""
            instruction = raw_instruction
            if instruction_type == "contextual":
                if match := CONTEXT_RE.search(raw_instruction):
                    context = match.group(2)
                    instruction = match.group(1)
                else:
                    continue
            else:
                instruction = re.sub(
                    r"^\s*\d+\s*\.\s*", "", self.clean_instruction_text(raw_instruction)
                ).capitalize()
                if raw_instruction.rstrip().endswith(":"):
                    instruction += ":"
            if not instruction:
                continue
            if (
                not self.min_instruction_length
                < len(instruction.split())
                < self.max_instruction_length
            ):
                continue
            # Skip various prompts that have been deemed unsuitable for language models
            # by the self-instruct team.
            if (
                self.skip_instruction_re.search(instruction)
                or self.code_gen_re.search(instruction)
                or instruction[0] in string.punctuation
                or not instruction[0].isascii()
            ):
                continue
            instructions.append(
                {
                    "instruction": instruction,
                    "context": context,
                    "instruction_type": instruction_type,
                }
            )
        return instructions

    @backoff.on_exception(backoff.expo, (RateLimitError, TooManyRequestsError))
    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a post request to OpenAI API.

        :param path: URL path to send request to.
        :type path: str

        :param payload: Dict containing request body/payload.
        :type payload: Dict[str, Any]

        :return: Response object.
        :rtype: Dict[str, Any]
        """
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        async with aiohttp.ClientSession() as session:
            result = await session.post(
                f"{OPENAI_API_BASE_URL}{path}", headers=headers, json=payload
            )
            if result.status != 200:
                text = await result.text()
                if "too many requests" in text.lower():
                    raise TooManyRequestsError(text)
                if "rate limit reached" in text.lower():
                    raise RateLimitError(text)
                elif "context_length_exceeded" in text.lower():
                    raise ContextLengthExceededError(text)
                else:
                    raise BadResponseError(text)
            result = await result.json()
        logger.debug(f"POST {path} with {json.dumps(payload)}: {json.dumps(result)}")
        self.used_tokens += result["usage"]["total_tokens"]
        if self.max_usage_tokens and self.used_tokens > self.max_usage_tokens:
            raise TokensExhaustedError(f"Max token usage exceeded: {self.used_tokens}")
        logger.debug(f"token usage: {self.used_tokens}")
        return result

    async def _post_no_exc(self, *a, **k):
        """Post, ignoring all exceptions."""
        try:
            return await self._post(*a, **k)
        except Exception as ex:
            logger.error(f"Error performing post: {ex}")
            import traceback

            print(traceback.format_exc())
        return None

    async def _generate_instruction_batch(
        self, instruction_count: int
    ) -> List[Dict[str, Any]]:
        """Generate an set of instructions.  Wrapped by generate_instruction_batch."""
        instruction_type = "open"
        m_target_list = self.open_machine_tasks
        s_target_list = self.open_seed_tasks
        rand = random.random()
        if rand <= self.classification_ratio:
            instruction_type = "classification"
            m_target_list = self.classification_machine_tasks
            s_target_list = self.classification_seed_tasks
        elif rand <= self.classification_ratio + self.contextual_ratio:
            instruction_type = "contextual"
            m_target_list = self.contextual_machine_tasks
            s_target_list = self.contextual_seed_tasks

        instructions = []
        if m_target_list:
            instructions = random.sample(m_target_list, min(2, len(m_target_list)))
        instructions += random.sample(
            s_target_list, instruction_count - len(instructions)
        )
        random.shuffle(instructions)
        prompt = self.generate_prompt_from_instructions(instructions, instruction_type)

        path = "/v1/completions" if self._completions else "/v1/chat/completions"
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self._completions:
            payload["prompt"] = prompt
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        new_instructions = self.extract_instructions_from_response(
            await self._post(path, payload), instruction_type
        )
        futures = []
        for instruction in new_instructions:
            get_response_prompt = instruction["instruction"]
            if instruction.get("context", "").strip():
                get_response_prompt += "\n" + instruction["context"]
            new_payload = copy.deepcopy(payload)
            if self._completions:
                new_payload["prompt"] = get_response_prompt
            else:
                new_payload["messages"][0]["content"] = get_response_prompt
            futures.append(self._post_no_exc(path, new_payload))
        results = await asyncio.gather(*futures)
        instructions_with_responses = []
        for idx, result in enumerate(results):
            if not result:
                logger.warning(
                    f"Error generating response for machine generated instruction: {new_instructions[idx]}"
                )
                continue
            try:
                response = (
                    result["choices"][0]["text"]
                    if self._completions
                    else result["choices"][0]["message"]["content"]
                )
                if response.strip():
                    logger.debug(
                        "\n".join(
                            [
                                f"Generated instruction [type={new_instructions[idx]['instruction_type']}]:",
                                f"Prompt: {new_instructions[idx]['instruction']}",
                            ]
                            + (
                                [f"Context: {new_instructions[idx]['context']}"]
                                if new_instructions[idx]["context"]
                                else []
                            )
                            + [f"Response: {response.strip()}"]
                        )
                    )
                    instructions_with_responses.append(
                        {
                            **new_instructions[idx],
                            **{"response": response.strip()},
                        }
                    )
            except Exception as exc:
                logger.error(
                    f"Error parsing response for machine-generated isntruction: {exc}"
                )
        return instructions_with_responses

    async def generate_instruction_batch(self) -> List[Dict[str, Any]]:
        """Generate a batch of instructions.

        :return: List of machine-generated instructions with responses.
        :rtype: List[Dict[str, Any]]
        """
        current_count = self.instructions_per_request
        while current_count > 3:
            try:
                return await self._generate_instruction_batch(current_count)
            except ContextLengthExceededError:
                current_count -= 1
        logger.error("Couldn't generate instruction batch due to context length error")
        return []

    async def run(self):
        """Run the self-instruct, instruction generation task to completion."""
        if os.path.exists(self.output_path) and not self.overwrite:
            raise RuntimeError(
                f"Output path: {self.output_path} already exists, overwrite false"
            )
        with open(self.output_path, "w") as outfile:
            while len(self.machine_tasks) <= self.instruction_count:
                futures = [self.generate_instruction_batch() for _ in range(10)]
                results = None
                try:
                    results = await asyncio.gather(*futures)
                except TokensExhaustedError:
                    logger.error("Max token usage reached.")
                    break
                new_instructions = [inst for insts in results for inst in insts]
                for inst in new_instructions:
                    self.machine_tasks.append(inst)
                    if inst["instruction_type"] == "classification":
                        self.classification_machine_tasks.append(inst)
                    elif inst.get("context", "").strip():
                        self.contextual_machine_tasks.append(inst)
                    else:
                        self.open_machine_tasks.append(inst)
                    outfile.write(json.dumps(inst))
                logger.info(
                    f"Generated {len(self.machine_tasks)} of {self.instruction_count}, tokens used: {self.used_tokens}"
                )
        logger.success(
            f"Finished generating {len(self.machine_tasks)} instructions and responses."
        )


def main():
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    instructor = SelfInstructor(**vars(parser.parse_args()))
    asyncio.run(instructor.run())


if __name__ == "__main__":
    main()
