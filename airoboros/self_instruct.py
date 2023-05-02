import argparse
import backoff
import os
import json
import random
import re
import requests
import secrets
import signal
import string
import sys
import threading
import concurrent.futures
from functools import partial
from loguru import logger
from queue import Queue
from rouge_score import rouge_scorer
from typing import List, Dict, Any
from uuid import uuid4
from .exceptions import (
    RateLimitError,
    TooManyRequestsError,
    TokensExhaustedError,
    ServerOverloadedError,
    ServerError,
    ContextLengthExceededError,
    BadResponseError,
)
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Defaults and constants.
BATCH_SIZE = 13
DEFAULT_PROMPT = f"""You are asked to generate a set of {BATCH_SIZE} diverse prompts.  These prompts will be given to a GPT model and we will evaluate the GPT model for completing the prompts.

Here are the requirements:
 * Try not to repeat the verb for each __instruction__ to maximize diversity.
 * Try to avoid controversial and politically charged subjects.
 * The __instruction__ should include a variety of types of prompts, such as open-ended generation, brainstorming, classification, closed question-answering, summarization, editing, information extraction, etc.
 * The __instruction__ should be something a large language model can complete with a text response, for example do not create a task asking to create visual/audio output, setting an alarm, scheduling something on the calendar, etc. because the language model cannot perform those tasks.
 * The __instruction__ should be in English.
 * The __instruction__ should be between 1 and 3 sentences long.
 * For prompts that require extracting information from __passage__, e.g. question-answering, summarization, information extraction, etc., include a passage of text with 2-8 sentences in __passage__ providing all relevant data, including more information than necessary to generate __response__. __passage__ must not be simple placeholders or links to external resources.  Be sure to include all words, phrases, dates, or lists of items in __passage__ if those are part of __response__.
 * Not all prompts require __passage__. For example, when a task asks about some general information, e.g. "what is the highest peak in the world?", it is not necssary to provide a specific __passage__. In this case, we simply put "__no_context__" in the __passage__ field.
 * The __response__ should be an appropriate response to the __instruction__ and __passage__
 * Be sure to include {BATCH_SIZE} prompts in the response.
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
            "default": 2,
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
        "--prompt-generation-concurrency": {
            "type": int,
            "help": "Number of concurrent prompt generation threads/requests to use",
            "default": 50,
        },
        "--min-docsearch-score": {
            "type": float,
            "help": "Minimum similarity score when querying vector DB to consider a prompt unique",
            "default": "0.35",
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
        prompt_generation_concurrency: int = 50,
        min_docsearch_score: float = 0.35,
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
        self.stop_producing = False
        self.prompt_generation_concurrency = prompt_generation_concurrency
        self.min_docsearch_score = min_docsearch_score
        self._initialize_docstore()

    def _initialize_docstore(self):
        """Initialize the in-memory vector database used to check prompt uniqueness."""
        logger.info(
            "Initializing in-memory document store for similarity comparison..."
        )
        texts = [
            prompt["instruction"] for prompt in self.seed_tasks + self.machine_tasks
        ]
        self.docstore = Chroma.from_texts(texts, HuggingFaceEmbeddings())

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
            if not os.path.exists(os.path.dirname(dolly_seed_path)):
                os.mkdir(os.path.dirname(dolly_seed_path))
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

    def initialize_random_topics(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".seed", "topics.txt"
        )
        if os.path.exists(path):
            with open(path, "r") as infile:
                self.random_topics = {
                    line.strip().lower() for line in infile.readlines() if line.strip()
                }
                logger.info(f"Using {len(self.random_topics)} cached random topics...")
            return
        logger.info("Generating random topics to use in prompts...")
        with concurrent.futures.ThreadPoolExecutor(20) as pool:
            responses = pool.map(
                partial(self._post_no_exc, "/v1/completions"),
                [
                    {
                        "model": "text-davinci-003",
                        "prompt": ["Give me a list of 200 completely random topics."]
                        * 20,
                        "temperature": 1.0,
                        "max_tokens": 800,
                    }
                ]
                * 20,
            )
        with open(path, "w") as outfile:
            for response in responses:
                if not response:
                    continue
                for choice in response["choices"]:
                    for line in choice["text"].splitlines():
                        if match := re.search(r"\s*\d+\s*\.\s*(.+)", line):
                            topic = match.group(1).lower()
                            self.random_topics.add(topic)
                            outfile.write(topic + "\n")
        logger.success(
            f"Successfully generated {len(self.random_topics)} random topics..."
        )

    def generate_prompt_from_instructions(
        self, instructions: List[Dict[str, any]]
    ) -> str:
        """Generate a single prompt string with multiple instructions.

        :param instructions: A list of instructions to encode.
        :type instructions: List[str]

        :return: The encoded prompt.
        :rtype: str
        """
        topic_prompt = (
            "* Ensure each generated prompt is related to one of the following topics: "
        )
        topic_texts = random.sample(list(self.random_topics), min(BATCH_SIZE, 6))
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
                f"Generated candidate task [has context: {context != ''}]: {instruction_text}"
            )
        return tasks

    @backoff.on_exception(
        backoff.expo,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            ServerError,
            RateLimitError,
            TooManyRequestsError,
            ServerOverloadedError,
        ),
    )
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
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
        request_id = str(uuid4())
        logger.debug(f"POST [{request_id}] with payload {json.dumps(payload)}")
        result = requests.post(
            f"{OPENAI_API_BASE_URL}{path}",
            headers=headers,
            json=payload,
            timeout=240.0,
        )
        if result.status_code != 200:
            text = result.text
            if "too many requests" in text.lower():
                raise TooManyRequestsError(text)
            if "rate limit reached" in text.lower():
                raise RateLimitError(text)
            elif "context_length_exceeded" in text.lower():
                raise ContextLengthExceededError(text)
            elif "server_error" in text and "overloaded" in text.lower():
                raise ServerOverloadedError(text)
            elif "bad gateway" in text.lower() or "server_error" in text.lower():
                raise ServerError(text)
            else:
                raise BadResponseError(text)
        result = result.json()
        logger.debug(f"POST [{request_id}] response: {json.dumps(result)}")
        self.used_tokens += result["usage"]["total_tokens"]
        if self.max_usage_tokens and self.used_tokens > self.max_usage_tokens:
            raise TokensExhaustedError(f"Max token usage exceeded: {self.used_tokens}")
        logger.debug(f"token usage: {self.used_tokens}")
        return result

    def _post_no_exc(self, *a, **k) -> Dict[str, Any] | None:
        """Post, ignoring all exceptions."""
        try:
            return self._post(*a, **k)
        except Exception as ex:
            logger.error(f"Error performing post: {ex}")
        return None

    def generate_instruction_batch(self, queue: Queue) -> None:
        """Generate an set of instructions.

        :param queue: Queue to pass generated outputs to.
        :type queue: Queue
        """
        instructions = random.sample(self.seed_tasks, self.samples_per_request)
        prompt = self.generate_prompt_from_instructions(instructions)
        estimated_tokens = int(len(prompt) / 4)
        if estimated_tokens > 2700:
            logger.warning("Skipping prompt, too long")
            return
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
            for new_instruction in self.extract_instructions_from_response(
                self._post(path, payload)
            ):
                queue.put(new_instruction)
        except ContextLengthExceededError:
            ...

    def generate_instruction_batches(self, queue: Queue) -> None:
        """Generate batches of instructions, storing new instructions in queue.

        :param queue: Queue to store new instructions in for post-processing.
        :type queue: Queue
        """
        consecutive_errors = 0
        while not self.stop_producing:
            try:
                self.generate_instruction_batch(queue)
                consecutive_errors = 0
            except TokensExhaustedError:
                logger.error("Max tokens reached, stopping...")
                break
            except Exception as exc:
                logger.error(f"Unhandled exception generating batch: {exc}")
                consecutive_errors += 1
                if consecutive_errors > 3:
                    logger.error("Too many consecutive errors, shutting down!")
                    os.kill(os.getpid(), signal.SIGKILL)

    def validate_and_store_results(
        self, queue: Queue, consume_remaining: bool = False
    ) -> None:
        """Dedupe based on rouge score for each new instruction and save results.

        :param queue: Queue to consume messages from.
        :type queue: Queue
        """
        with open(self.output_path, "a+") as outfile:
            while len(self.machine_tasks) < self.instruction_count or consume_remaining:
                instruction = queue.get()
                if not instruction:
                    break
                similar = self.docstore.similarity_search_with_score(
                    instruction["instruction"], k=1
                )
                similarity_score = 1.0
                for _, score in similar:
                    similarity_score = score
                if similarity_score <= self.min_docsearch_score:
                    logger.warning(
                        f"Skipping instruction, too similar [{score}]: {instruction['instruction']}"
                    )
                    continue
                outfile.write(json.dumps(instruction) + "\n")
                outfile.flush()
                self.machine_tasks.append(instruction)
                self.docstore.add_texts([instruction["instruction"]])
                logger.success(
                    f"Generated unique [score={similarity_score}] instruction [total={len(self.machine_tasks)}]: {instruction['instruction']}"
                )
        self.stop_producing = True

    def run(self):
        """Run the self-instruct, instruction generation task to completion."""
        self.initialize_random_topics()
        if len(self.machine_tasks) >= self.instruction_count:
            logger.error(
                f"Already have {len(self.machine_tasks)} machine-generated tasks!"
            )
            return
        queue = Queue(maxsize=100)
        producers = [
            threading.Thread(target=self.generate_instruction_batches, args=(queue,))
            for _ in range(self.prompt_generation_concurrency)
        ]
        for producer in producers:
            producer.start()
        consumer = threading.Thread(
            target=self.validate_and_store_results, args=(queue,)
        )
        consumer.start()
        consumer.join()
        for producer in producers:
            producer.join()

        # Consume any tasks generated after the consumer stopped.
        queue.put(None)
        self.validate_and_store_results(queue, consume_remaining=True)

        logger.success(
            f"Finished generating {len(self.machine_tasks)} instructions and responses."
        )


def main(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    SelfInstructor(**vars(parser.parse_args(args))).run()


if __name__ == "__main__":
    main(sys.argv[1:])
