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
CONTEXTUAL_PROMPT = f"""Create a few instructions that can be provided to a GPT system to create text and a task related to the text.  Use diverse verbs, subject matters, and writing styles, and don't use any placeholders.

Examples:
 * Generate a few paragraphs about the process of making damascus steel.
 * Write a short story about a goat/bird chimera that enjoys sailing.
 * Compose a news article about a breakthrough in tire technology.
 * Give me a detailed description of the Battle of Rennell Island during World War II, along with key dates, locations, and people involved.

Ensure each instruction is related to one of the following topics:
{{topics}}

Numbered list of {BATCH_SIZE} instructions:
"""
CONTEXT_TASK_INJECTION = "At the beginning of the text, add an instruction that can make use the text to respond.  This instruction type can be closed-context question answering, summarization, information extraction, etc."
DEFAULT_PROMPT = f"""Create a set of {BATCH_SIZE} diverse instructions.

Requirements for the instructions:
 * Do not repeat the verb for each instruction to maximize diversity.
 * Try to avoid controversial and politically charged subjects.
 * The list of instructions should include a variety of types of prompts, such as open-ended text generation, creative writing, brainstorming, classification, editing, logical reasoning, etc.
 * Each instruction must be something a large language model can complete with a text response, for example do not create a task asking to create visual/audio output, setting an alarm, scheduling something on the calendar, etc. because the language model cannot perform those tasks.
 * Each instruction should be in English, and be between 1 and 3 sentences long.
 * Do not include any prompts that would require additional information, for example instructions to summarize or extract information from a passage of text.
 * Any instruction referencing a list of objects, such as classifying a list of items, should include the list of items.

Ensure each instruction is related to one of the following topics:
{{topics}}

Numbered list of {BATCH_SIZE} prompts:
"""
SKIP_WORDS = ["image", "graph", "picture", "file", "map", "draw", "plot", "go to"]
SKIP_SEARCH_RE = re.compile(r"\b{'|'.join(SKIP_WORDS)}s?\b", re.I)
CODE_GEN_RE = re.compile(
    r"^(?:write|generate|create)(?:a )?(?:\w+ )?(?:script|program|code)\W", re.I
)
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
        "--output-path": {
            "type": str,
            "help": "path to store all generated instructions in",
            "default": "instructions.jsonl",
        },
        "--topics-path": {
            "type": str,
            "help": "path to a newline separated list of topics",
            "default": "topics.txt",
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
            "help": "prompt prefix to use for generating non-contextual instructions",
        },
        "--contextual-prompt": {
            "type": str,
            "default": CONTEXTUAL_PROMPT,
            "help": "prompt to use for generating contextual prompts",
        },
        "--contextual-prompt-ratio": {
            "type": float,
            "default": 0.15,
            "help": "ratio of prompts that should be contextual, e.g. summarization of an article",
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
        "--temperature": {
            "type": float,
            "default": 0.9,
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
        output_path: str = "instructions.jsonl",
        topics_path: str = "topics.txt",
        overwrite: bool = False,
        append: bool = True,
        prompt: str = DEFAULT_PROMPT,
        contextual_prompt: str = CONTEXTUAL_PROMPT,
        contextual_prompt_ratio: float = 0.15,
        skip_instruction_re: re.Pattern = SKIP_SEARCH_RE,
        code_gen_re: re.Pattern = CODE_GEN_RE,
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
        self.output_path = os.path.abspath(output_path)
        self.topics_path = os.path.abspath(topics_path)
        self.overwrite = overwrite
        self.append = append
        self.prompt = prompt
        self.contextual_prompt = contextual_prompt
        self.contextual_prompt_ratio = contextual_prompt_ratio
        self.skip_instruction_re = skip_instruction_re
        if isinstance(skip_instruction_re, str):
            self.skip_instruction_re = re.compile(skip_instruction_re, re.I)
        self.code_gen_re = code_gen_re
        if isinstance(code_gen_re, str):
            self.code_gen_re = re.compile(code_gen_re, re.I)
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_usage_tokens = max_usage_tokens
        self.validate_model()
        self.used_tokens = 0
        self.random_topics = set([])
        self.stop_producing = False
        self.prompt_generation_concurrency = prompt_generation_concurrency
        self.min_docsearch_score = min_docsearch_score
        self.initialize_docstores()

    def initialize_docstores(self):
        """Initialize the in-memory vector databases used to check prompt uniqueness."""
        self.machine_task_count = 0
        docs = []
        if os.path.exists(self.output_path):
            if self.overwrite:
                os.remove(self.output_path)
            elif self.append:
                with open(self.output_path, "r") as infile:
                    for line in infile.readlines():
                        task = json.loads(line)
                        self.machine_task_count += 1
                        docs.append(task["instruction"])
                logger.info(
                    f"Found {self.machine_task_count} existing machine-generated instructions."
                )
            else:
                raise RuntimeError(
                    f"{self.output_path} already exists, but overwrite and append are false!"
                )
        logger.info(
            "Initializing in-memory document store for similarity comparison..."
        )
        if not docs:
            docs = ["__initialize__"]
        embeddings = HuggingFaceEmbeddings()
        self.docstore = Chroma.from_texts(docs, embeddings)
        self.pending_docstore = Chroma.from_texts(["__initialize__"], embeddings)

    def validate_model(self):
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

    def initialize_random_topics(self):
        if os.path.exists(self.topics_path):
            with open(self.topics_path, "r") as infile:
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
                        "prompt": [
                            "Give me a numbered list of 200 completely random topics."
                        ]
                        * 20,
                        "temperature": 1.0,
                        "max_tokens": 3000,
                    }
                ]
                * 20,
            )
        with open(self.topics_path, "w") as outfile:
            for response in responses:
                if not response:
                    continue
                for choice in response["choices"]:
                    for line in choice["text"].splitlines():
                        if match := re.search(r"\s*\d+\s*\.\s*(.+)", line):
                            topic = match.group(1).lower().strip()
                            if not topic:
                                continue
                            self.random_topics.add(topic)
                            outfile.write(topic + "\n")
        logger.success(
            f"Successfully generated {len(self.random_topics)} random topics..."
        )

    def generate_prompt(self, template: str):
        """Generate a single prompt, inserting random topics.

        :param template: The prompt template to use.
        :type template: str

        :return: The prompt, including a list of random topics.
        :rtype: str
        """
        topics = "\n".join(
            [
                f" * {topic}"
                for topic in random.sample(list(self.random_topics), min(BATCH_SIZE, 7))
            ]
        )
        return template.format(topics=topics)

    def is_too_similar(self, instruction, docstore) -> bool:
        """Check if the input instruction is too similar to an existing instruction.

        :param instruction: The candidate instruction to check.
        :type instruction: str

        :param docstore: The docstore to perform similarity search against.
        :type docstore: Chroma

        :return: True if too similar, otherwise false.
        :rtype: bool
        """
        for _, score in docstore.similarity_search_with_score(instruction, k=1):
            if score < self.min_docsearch_score:
                return True
        return False

    def extract_instructions_from_response(self, text: str) -> List[str]:
        """Extract the list of instructions from the OpenAI response.

        :param text: The OpenAI response text.
        :type text: str

        :return: List of instructions.
        :rtype: List[str]
        """
        if not text:
            return []
        instructions = []
        for instruction in re.findall(r"\s*\d+\s*\.\s(.*)\s*", text):
            # Skip various prompts that have been deemed unsuitable for language models
            # by the self-instruct team.
            if (
                self.skip_instruction_re.search(instruction)
                or self.code_gen_re.search(instruction)
                or instruction[0] in string.punctuation
                or not instruction[0].isascii()
            ):
                logger.warning(
                    f"Skipping instruction: {instruction} [code, ascii, other unsuitable]"
                )
                continue
            if self.is_too_similar(instruction, self.docstore) or self.is_too_similar(
                instruction, self.pending_docstore
            ):
                logger.warning(f"Skipping instruction, too similar: {instruction}")
            self.pending_docstore.add_texts([instruction])
            instructions.append(instruction)
            logger.info(f"Generated candidate task: {instruction}")
        return instructions

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
            timeout=600.0,
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

    def generate_response(self, instruction: str) -> str:
        """Call OpenAI with the specified instruction and return the text response.

        :param instruction: The instruction to respond to.
        :type instruction: str

        :return: Response text.
        :rtype: str
        """
        path = "/v1/completions" if self._completions else "/v1/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self._completions:
            payload["prompt"] = instruction
            payload["max_tokens"] = 4000
        else:
            payload["messages"] = [{"role": "user", "content": instruction}]
        response = self._post_no_exc(path, payload)
        if (
            not response
            or not response.get("choices")
            or response["choices"][0]["finish_reason"] == "length"
        ):
            return None
        text = None
        if self._completions:
            text = response["choices"][0]["text"]
        else:
            text = response["choices"][0]["message"]["content"]
        return text

    def generate_instruction_batch(self, queue: Queue) -> None:
        """Generate an set of instructions.

        :param queue: Queue to pass generated outputs to.
        :type queue: Queue
        """
        contextual = random.random() <= self.contextual_prompt_ratio
        prompt = self.generate_prompt(
            self.prompt if not contextual else self.contextual_prompt
        )
        for new_instruction in self.extract_instructions_from_response(
            self.generate_response(prompt)
        ):
            prompt = new_instruction
            if contextual:
                prompt = self.generate_response(
                    "  ".join([new_instruction, CONTEXT_TASK_INJECTION])
                )
            if prompt:
                response = self.generate_response(prompt)
                if response:
                    queue.put({"instruction": prompt, "response": response})
                else:
                    logger.error(
                        f"Error generating response to: {prompt.splitlines()[0]}..."
                    )

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
            while self.machine_task_count < self.instruction_count or consume_remaining:
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
                self.machine_task_count += 1
                self.docstore.add_texts([instruction["instruction"]])
                logger.success(
                    f"Generated unique [score={similarity_score}] instruction [total={self.machine_task_count}]: {instruction['instruction']}"
                )
        self.stop_producing = True

    def run(self):
        """Run the self-instruct, instruction generation task to completion."""
        self.initialize_random_topics()
        if self.machine_task_count >= self.instruction_count:
            logger.error(
                "Already have {self.machine_task_count} machine-generated tasks!"
            )
            return
        queue = Queue()
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
            f"Finished generating {self.machine_task_count} instructions and responses."
        )


def main(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    SelfInstructor(**vars(parser.parse_args(args))).run()


if __name__ == "__main__":
    main(sys.argv[1:])
