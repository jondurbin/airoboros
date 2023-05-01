import asyncio
import aiohttp
import argparse
import backoff
import os
import json
import random
import re
import requests
import secrets
import string
from multiprocessing import Pool
from functools import partial
from loguru import logger
from typing import List, Dict, Any
from rouge_score import rouge_scorer
from .exceptions import (
    RateLimitError,
    TooManyRequestsError,
    TokensExhaustedError,
    ServerOverloadedError,
    ContextLengthExceededError,
    BadResponseError,
)

# Defaults and constants.
BATCH_SIZE = 13
DEFAULT_PROMPT = f"""You are asked to generate a set of {BATCH_SIZE} diverse prompts.  These prompts will be given to a GPT model and we will evaluate the GPT model for completing the prompts.

Here are the requirements:
 * Try not to repeat the verb for each __instruction__ to maximize diversity.
 * Try to avoid constroversial and politically charged subjects.
 * The __instruction__ should include a variety of types of prompts, such as open-ended generation, brainstorming, classification, closed question-answering, summarization, editing, information extraction, etc.k
 * The __instruction__ should be something a large language model can complete with a text response, for example do not create a task asking to create visual/audio output, setting an alarm, scheduling something on the calendar, etc. because the language model cannot perform those tasks.
 * The __instruction__ should be in English.
 * The __instruction__ should be 1 to 2 sentences long.
 * For prompts that require extracting information from __passage__, e.g. question-answering, summarization, information extraction, etc., include a passage of text with 2-8 sentences in __passage__ providing all relevant data, including more information than necessary to generate __response__. __passage__ must not be simple placeholders or links to external resources.  Be sure to include all words, phrases, dates, or lists of items in __passage__ if those are part of __response__.
 * Not all prompts require __passage__. For example, when a task asks about some general information, e.g. "what is the highest peak in the world?", it is not necssary to provide a specific __passage__. In this case, we simply put "__no_context__" in the __passage__ field.
 * The __response__ should be an appropriate response to the __instruction__ and __passage__
 * Be sure to include {BATCH_SIZE} propts in the response.
REPLACE_TOPICS

List of {BATCH_SIZE} prompts:
"""
SKIP_WORDS = ["image", "graph", "picture", "file", "map", "draw", "plot", "go to"]
SKIP_SEARCH_RE = re.compile(r"\b{'|'.join(SKIP_WORDS)}s?\b", re.I)
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
        "--append": {
            "action": "store_true",
            "help": "append to output path if it exists",
        },
        "--prompt": {
            "type": str,
            "default": DEFAULT_PROMPT,
            "help": "prompt prefix to use for generating tasks",
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
        "--samples-per-request": {
            "type": str,
            "default": 3,
            "help": "number of random sample instructions to include in prompts",
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
        append: bool = True,
        use_dolly_seed: bool = True,
        prompt: str = DEFAULT_PROMPT,
        skip_instruction_re: re.Pattern = SKIP_SEARCH_RE,
        code_gen_re: re.Pattern = CODE_GEN_RE,
        min_instruction_length: int = 3,
        max_instruction_length: int = 150,
        samples_per_request: int = 3,
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
        self.append = append
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.prompt = prompt
        self.skip_instruction_re = skip_instruction_re
        if isinstance(skip_instruction_re, str):
            self.skip_instruction_re = re.compile(skip_instruction_re, re.I)
        self.code_gen_re = code_gen_re
        if isinstance(code_gen_re, str):
            self.code_gen_re = re.compile(code_gen_re, re.I)
        self.min_instruction_length = min_instruction_length
        self.max_instruction_length = max_instruction_length
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.samples_per_request = samples_per_request
        self.max_usage_tokens = max_usage_tokens
        self._validate_model()
        self.machine_tasks = []
        self._initialize_seed_tasks(seed_tasks, seed_tasks_path, use_dolly_seed)
        self.used_tokens = 0
        self.random_topics = set([])

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
        logger.info(f"Found {len(self.seed_tasks)} human seed tasks to use...")
        if os.path.exists(self.output_path):
            if self.overwrite:
                os.remove(self.output_path)
            elif self.append:
                with open(self.output_path, "r") as infile:
                    self.machine_tasks = [
                        json.loads(line) for line in infile.readlines()
                    ]
                logger.info(
                    f"Found {len(self.machine_tasks)} pre-existing machine seed tasks to use..."
                )
            else:
                raise RuntimeError(
                    f"{self.output_path} already exists, but overwrite and append are false!"
                )

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

    async def initialize_random_topics(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".seed", "topics.txt"
        )
        if os.path.exists(path):
            with open(path, "r") as infile:
                self.random_topics = {
                    line.strip() for line in infile.readlines() if line.strip()
                }
                logger.info(f"Using {len(self.random_topics)} cached random topics...")
            return
        logger.info("Generating random topics to use in prompts...")
        futures = [
            self._post_no_exc(
                "/v1/completions",
                {
                    "model": "text-davinci-003",
                    "prompt": ["Give me a list of 100 completely random topics."] * 25,
                    "temperature": 1.0,
                    "max_tokens": 500,
                },
            )
            for _ in range(25)
        ]
        responses = await asyncio.gather(*futures)
        with open(path, "w") as outfile:
            for response in responses:
                if not response:
                    continue
                for choice in response["choices"]:
                    for line in choice["text"].splitlines():
                        if match := re.search(r"\s*\d+\s*\.\s*(.+)", line):
                            self.random_topics.add(match.group(1))
                            outfile.write(match.group(1) + "\n")
        logger.success(
            f"Successfully generated {len(self.random_topics)} random topics..."
        )

    async def generate_prompt_from_instructions(
        self, instructions: List[Dict[str, any]]
    ) -> str:
        """Generate a single prompt string with multiple instructions.

        :param instructions: A list of instructions to encode.
        :type instructions: List[str]

        :return: The encoded prompt.
        :rtype: str
        """
        topic_prompt = (
            "* Each __instruction__ must be related to one of the following topics: "
        )
        topic_texts = random.sample(list(self.random_topics), min(BATCH_SIZE, 5))
        topic_prompt += ", ".join(topic_texts)
        prompt = [self.prompt.replace("REPLACE_TOPICS", topic_prompt)]
        for idx, instruction in enumerate(instructions):
            text = self.clean_instruction_text(instruction["instruction"])
            prompt.append(f"{idx + 1}. __instruction__: {text}")
            context = (
                "__no_context__"
                if not instruction["context"].strip()
                else instruction["context"].strip()
            )
            prompt.append(f"{idx + 1}. __passage__: {context}")
            prompt.append(f"{idx + 1}. __response__: {instruction['response']}")
        return "\n".join(prompt)

    def extract_instructions_from_response(
        self, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract the list of instructions from the OpenAI response.

        :param response: The response from the OpenAI request.
        :type re: Dict[str, Any]

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

        tasks = []
        for instruction in re.findall(
            r"(\d+\s*\.\s*__instruction__:[\s\r\n]*.*?)(?=\d+\s*\.\s*__|$)",
            text,
            re.DOTALL,
        ):
            idx = instruction.split(".")[0].strip()
            instruction_text = instruction.split("__instruction__:")[-1].strip()
            if not instruction_text:
                continue
            if (
                not self.min_instruction_length
                < len(instruction_text.split())
                < self.max_instruction_length
            ):
                logger.warning(
                    f"Skipping instruction: {instruction_text} [instruction length]"
                )
            # Skip various prompts that have been deemed unsuitable for language models
            # by the self-instruct team.
            if (
                self.skip_instruction_re.search(instruction_text)
                or self.code_gen_re.search(instruction_text)
                or instruction_text[0] in string.punctuation
                or not instruction_text[0].isascii()
            ):
                logger.warning(
                    f"Skipping instruction: {instruction_text} [code, ascii, other unsuitable]"
                )
                continue
            context = re.search(
                f"(?<!\\d){idx}\\s*\\.\\s*__passage__:[\\r\\n\\s]*(.*?)(?=\\d+\\s*\\.\\s*__|$)",
                text,
                re.DOTALL,
            )
            if not context:
                logger.warning(
                    f"Skipping instruction: {instruction_text} [context not provided]"
                )
                continue
            context = context.group(1).strip()
            if context == "__no_context__":
                context = ""
            response = re.search(
                f"(?<!\\d){idx}\\s*\\.\\s*__response__:[\\r\\n\\s]*(.*?)(?=\\d+\\s*\\.\\s*__|$)",
                text,
                re.DOTALL,
            )
            if not response or not response.group(1).strip():
                logger.warning(
                    f"Skipping instruction: {instruction_text} [response not provided]"
                )
                continue
            response = response.group(1).strip()
            if len(response) > len(context):
                logger.warning("Ignoring context, shorter than response.")
                context = ""
            else:
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
                if scorer.score(context, response)["rougeL"].fmeasure >= 0.7:
                    logger.warning("Ignoring context, too similar to response.")
                    context = ""
            tasks.append(
                {
                    "instruction": instruction_text,
                    "context": context,
                    "response": response,
                }
            )
            logger.info(
                f"Generated new task [has context: {context != ''}]: {instruction_text}"
            )
        return tasks

    @backoff.on_exception(
        backoff.expo, (RateLimitError, TooManyRequestsError, ServerOverloadedError)
    )
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
                elif "server_error" in text and "overloaded" in text.lower():
                    raise ServerOverloadedError(text)
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
        return None

    async def generate_instruction_batch(self):
        """Generate an set of instructions.

        :return: List of machine-generated instructions, contexts, and responses.
        :rtype: List[Dict[str, Any]]
        """
        instructions = random.sample(self.seed_tasks, self.samples_per_request)
        prompt = await self.generate_prompt_from_instructions(instructions)
        estimated_tokens = int(len(prompt) / 4)
        if estimated_tokens > 2700:
            logger.warning("Skipping prompt, too long")
            return []
        path = "/v1/completions" if self._completions else "/v1/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": [f"{BATCH_SIZE+1}."],
            "max_tokens": 4000 - estimated_tokens if self._completions else None,
        }
        if self._completions:
            payload["prompt"] = prompt
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        try:
            return self.extract_instructions_from_response(
                await self._post(path, payload)
            )
        except ContextLengthExceededError:
            ...
        return []

    async def run(self):
        """Run the self-instruct, instruction generation task to completion."""
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        await self.initialize_random_topics()
        with open(self.output_path, "a+") as outfile:
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
                    with Pool(16) as pool:
                        scores = pool.map(
                            partial(scorer.score, inst["instruction"]),
                            [
                                item["instruction"]
                                for item in self.seed_tasks + self.machine_tasks
                            ],
                        )
                    scores = [score["rougeL"].fmeasure for score in scores]
                    max_score = max(scores)
                    if max_score > 0.7:
                        logger.warning(
                            f"Skipping instruction, too similar: {max_score}: {inst['instruction']}"
                        )
                        continue
                    self.machine_tasks.append(inst)
                    outfile.write(json.dumps(inst) + "\n")
                outfile.flush()
                logger.info(
                    f"Generated {len(self.machine_tasks)} of {self.instruction_count}, tokens used: {self.used_tokens}"
                )
        logger.success(
            f"Finished generating {len(self.machine_tasks)} instructions and responses."
        )


def main():
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    instructor = SelfInstructor(**vars(parser.parse_args()))
    asyncio.run(instructor.run())


if __name__ == "__main__":
    main()
