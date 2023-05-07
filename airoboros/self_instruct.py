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
DEFAULT_PROMPT = f"""You are asked to generate a set of {BATCH_SIZE} diverse instructions.  These instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
 * Try not to repeat the verb for each __instruction__ to maximize diversity.
 * Try to avoid controversial and politically charged subjects.
 * The __instruction__ should include a variety of types of instructions, such as open-ended text generation, creative writing, brainstorming, classification, contextual question-answering, summarization, editing, information extraction, logical reasoning, etc.
 * The __instruction__ should be something a large language model can complete with a text response, for example do not create a task asking to create or use visual/audio output, setting an alarm, scheduling something on the calendar, etc. because the language model cannot perform those tasks.
 * The __instruction__ should be in English.
 * If __instruction__ relates to a list of items, be sure to include the list of items in the __instruction__.
 * Do not include a response to the instructions, only the instructions themselves.
 * Be sure to include {BATCH_SIZE} instructions in the response.
REPLACE_TOPICS

List of {BATCH_SIZE} instructions:
"""
ENRICH_PROMPT = """You are a system designed to help users create fully encapsulated instructions to use in a one-shot, in-context learning GPT system.  The target GPT system cannot perform N-shot tasks and has no ability to ask for more information.

Here are the rules:
1. If the original user input would not require any additional information or context from the user, return the single word "sufficient"
2. If a GPT system would be able to generate a response to the instruction from common knowledge, return the single word "sufficient":
3. Return the single word "sufficient" for any inputs that are related to creative writing tasks, for example: writing a song, writing a story, writing a fictional review, brainstorming, imagination, etc.
4. If none of the previous rules apply, update the original user input to include a few paragraphs of text that are a reasonable facsimile of what a GPT system would expect a user to provide, for example pasting the content of a news article, wikipedia post, blog post, social media thread, list of items to classify, etc.
5. Do not generate a response to the original user input, only respond with either the single word "sufficient" or the updated prompt according to rule 4.

Example 1:
===
Original user input: Write a poem about a Hippo named Billy.

Response: sufficient

Example 2:
===
Original user imput: Summarize the key points in the article about LinkedIn.

Response: Summarize the key points from the following article regarding LinkedIn.

LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003.[5] It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals.[6] Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories.[5]

LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more.[7]


Original user input: {instruction}

Response:"""
SKIP_WORDS = ["image", "graph", "picture", "file", "map", "draw", "plot", "go to"]
SKIP_SEARCH_RE = re.compile(r"\b{'|'.join(SKIP_WORDS)}s?\b", re.I)
CODE_GEN_RE = re.compile(
    r"^(?:write|generate|create)(?:a )?(?:\w+ )?(?:script|program|code)\W", re.I
)
DOLLY_SEED_URL = "https://storage.googleapis.com/airoboros-dump/databricks-dolly-15k.jsonl"
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
        "--timeout": {
            "type": float,
            "help": "OpenAI request timeout",
            "default": 120.0,
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
        samples_per_request: int = 3,
        temperature: float = 0.7,
        top_p: float = 0.5,
        frequency_penalty: int = 0,
        presence_penalty: int = 2,
        max_usage_tokens: int | None = None,
        prompt_generation_concurrency: int = 50,
        min_docsearch_score: float = 0.35,
        timeout: float = 120.0,
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
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.samples_per_request = samples_per_request
        self.max_usage_tokens = max_usage_tokens
        self._validate_model()
        self.machine_tasks = []
        self.seed_categories = {}
        self._initialize_seed_tasks(seed_tasks, seed_tasks_path, use_dolly_seed)
        self.used_tokens = 0
        self.random_topics = set([])
        self.stop_producing = False
        self.prompt_generation_concurrency = prompt_generation_concurrency
        self.min_docsearch_score = min_docsearch_score
        self.timeout = timeout
        self._initialize_docstore()

    def _initialize_docstore(self):
        """Initialize the in-memory vector database used to check prompt uniqueness."""
        logger.info(
            "Initializing in-memory document store for similarity comparison..."
        )
        #texts = [
        #    prompt["instruction"] for prompt in self.seed_tasks + self.machine_tasks
        #]
        texts = ["foo"]
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
        for task in self.seed_tasks:
            if task["category"] not in self.seed_categories:
                self.seed_categories[task["category"]] = []
            self.seed_categories[task["category"]].append(task)
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
            text = instruction["instruction"].strip()
            if instruction.get("context", "").strip():
                text += f"\n{instruction['context']}"
            prompt.append(f"\n{idx + 1}. __instruction__: {text}")
        return "\n".join(prompt)

    def generate_response(self, instruction: str) -> str:
        """Given the instruction text, get a response to the instruction.

        :param instruction: The synthetic instruction to get a response to.
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
            payload["max_tokens"] = 2000
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

    def enrich_instruction(self, instruction: str) -> str:
        """Enrich a prompt with any missing context that a GPT system would need to respond.

        :param instruction: The original input instruction.
        :type instruction: str
        """
        result = self.generate_response(ENRICH_PROMPT.format(instruction=instruction))
        if result.strip().lower() == "sufficient":
            return instruction
        return result

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
        print(text)
        print("*" * 80)
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
            # Skip instructions with placeholders or links.
            if re.search(r"\[insert", instruction_text, re.I) or re.search("https?:/", instruction_text, re.I):
                logger.warning(f"Skipping instruction: {instruction_text} [placeholder or URL]")
                continue
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
            for _, score in self.docstore.similarity_search_with_score(instruction_text, k=1):
                if score <= self.min_docsearch_score:
                    logger.warning(
                        f"Skipping instruction, too similar [{score}]: {instruction_text}"
                    )
                    continue

            # Now, get the response.
            if enriched := self.enrich_instruction(instruction_text):
                if enriched != instruction_text:
                    logger.info(f"Enriched prompt: {enriched}")
                    instruction_text = enriched
            task = {"instruction": instruction_text}
            tasks.append(task)
            logger.info(f"Generated candidate task: {task}")
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
            timeout=self.timeout,
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
            elif "bad gateway" in text.lower() or "server_error" in text.lower() or "gateway timeout" in text.lower():
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
        except Exception as exc:
            print(f"QWERQWERQWER: {exc}")
            raise

    def generate_instruction_batches(self, queue: Queue, thread_idx: int) -> None:
        """Generate batches of instructions, storing new instructions in queue.

        :param queue: Queue to store new instructions in for post-processing.
        :type queue: Queue

        :param thread_idx: Thread index.
        :type thread_idx: int
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
            break
        logger.success(f"Producer [{thread_idx}] finished generating instructions.")

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
            if not consume_remaining:
                logger.success(f"Reached instruction count [{self.instruction_count}], will consume remaining once producers finish...")
        self.stop_producing = True

    def run(self):
        """Run the self-instruct, instruction generation task to completion."""
        self.initialize_random_topics()
        if len(self.machine_tasks) >= self.instruction_count:
            logger.error(
                f"Already have {len(self.machine_tasks)} machine-generated tasks!"
            )
            return
        queue = Queue()
        producers = [
            threading.Thread(target=self.generate_instruction_batches, args=(queue, i))
            for i in range(self.prompt_generation_concurrency)
        ]
        for producer in producers:
            producer.start()
        consumer = threading.Thread(
            target=self.validate_and_store_results, args=(queue,)
        )
        consumer.start()
        consumer.join()
        for idx, producer in enumerate(producers):
            producer.join()
            logger.info(f"Producer {idx} has finished...")

        # Consume any tasks generated after the consumer stopped.
        queue.put(None)
        logger.info("Consuming results created after instruction limit reached...")
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
