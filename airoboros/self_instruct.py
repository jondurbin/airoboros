import aiohttp
import argparse
import asyncio
import backoff
import datetime
import os
import json
import random
import re
import requests
import secrets
import sys
import yaml
from collections import defaultdict
from loguru import logger
from time import sleep
from typing import List, Dict, Any
from uuid import uuid4
from airoboros.exceptions import (
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
MAX_DOCSTORE_SIZE = 15000
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
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
    ],
}


class SelfInstructor:
    """Class and methods used to generate instructions, based on self-instruct paper/code."""

    CLI_ARGS = {
        # The updated code with several instructors has way too many options to support
        # as CLI args, so we just accept the config file path now.
        "--config-path": {
            "type": str,
            "default": "config.yaml",
            "help": "path to the airobors configuration file",
        },
    }

    def __init__(self, *, config_path: str = "config.yaml"):
        """Constructor."""
        self.used_tokens = 0
        self.config_path = config_path
        self.load_config()
        self.instructor_counts = defaultdict(int)
        self.docstore_lock = asyncio.Semaphore(1)

    def load_config(self):
        """Load an advanced configuration from a YAML file."""
        raw_config = self.raw_config = yaml.safe_load(open(self.config_path).read())
        self.model = raw_config.get("model") or "gpt-4"
        self.openai_api_key = raw_config.get("openai_api_key") or os.environ.get(
            "OPENAI_API_KEY"
        )
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable or openai_api_key must be provided"
            )
        self.organization_id = raw_config.get("organization_id")
        self.topics_path = raw_config.get("topics_path") or "topics.txt"
        self.output_path = raw_config.get("output_path") or "instructions.jsonl"
        self.overwrite = str(raw_config.get("overwrite")).lower() == "true"
        self.append = str(raw_config.get("append")).lower() == "true"
        self.topic_avoidance = raw_config.get("topic_avoidance", "")
        self.response_filters = []
        for val in raw_config.get("response_filters") or []:
            self.response_filters.append(re.compile(val, re.I))
        self.max_tokens = (
            int(raw_config["max_tokens"]) if raw_config.get("max_tokens") else None
        )
        self.min_docsearch_score = float(raw_config.get("min_docsearch_score") or 0.35)
        api_params = raw_config.get("api_params") or {}
        self.api_params = {
            "temperature": float(api_params.get("temperature") or 0.7),
            "top_p": float(api_params.get("top_p") or 0.5),
            "frequency_penalty": float(api_params.get("frequency_penalty") or 0.0),
            "presence_penalty": float(api_params.get("presence_penalty") or 2.0),
        }
        self.topic_prompt = raw_config["topic_prompt"].format(
            topic_avoidance=self.topic_avoidance
        )
        self.topic_request_count = int(raw_config.get("topic_request_count") or 20)
        self.default_count = 100
        if raw_config.get("default_count") is not None:
            self.default_count = int(raw_config["default_count"])
        self.default_batch_size = 5
        if raw_config.get("default_batch_size") is not None:
            self.default_batch_size = raw_config["default_batch_size"]
        self.language = raw_config.get("language") or "English"

        # Validate the model for each generator.
        self.instructors = raw_config.get("instructors")
        self.validate_model(self.model)
        valid_models = {self.model: True}
        for key, config in self.instructors.items():
            if config.get("model") and config["model"] not in valid_models:
                self.validate_model(config["model"])
                valid_models[config["model"]] = True

    def initialize_docstores(self):
        """Initialize the in-memory vector databases used to check prompt uniqueness."""
        docs = []
        if os.path.exists(self.output_path):
            if self.overwrite:
                result = input("Remove and overwrite {output_path} (Y/N)? ")
                if result.strip().lower() == "y":
                    os.remove(self.output_path)
                else:
                    raise RuntimeError("Overwrite aborted.")
            elif self.append:
                with open(self.output_path, "r") as infile:
                    for line in infile.readlines():
                        task = json.loads(line)
                        self.instructor_counts[task.get("category", "general")] += 1
                        docs.append(task["instruction"])
                logger.info(
                    f"Found {len(docs)} existing machine-generated instruction(s)."
                )
                for category, count in self.instructor_counts.items():
                    logger.info(f" * category {category}: {count}")
            else:
                raise RuntimeError(
                    f"{self.output_path} already exists, but overwrite and append are false!"
                )
        logger.info(
            "Initializing in-memory document store for similarity comparison..."
        )
        if not docs:
            docs = ["__initialize__"]
        self.embeddings = HuggingFaceEmbeddings()
        self.docstores = [Chroma.from_texts(docs, self.embeddings)]
        self.docstore_size = len(docs)
        self.docstore_rotated_at = 0
        self.topic_index = 0
        if self.docstore_size >= MAX_DOCSTORE_SIZE:
            logger.info("Initializing fresh docstore due to doc count...")
            self.docstore_size = 0
            self.docstores.append(
                Chroma.from_texts(["__initialize__"], self.embeddings)
            )

    def validate_model(self, model):
        """Ensure the specified model is available, and configure the endpoint
        to use accordingly (chat completions or completions).
        """
        if model in MODEL_ENDPOINTS["completions"]:
            self._completions = True
        elif model in MODEL_ENDPOINTS["chat_completions"]:
            self._completions = False
        else:
            raise ValueError(f"Model is not currently supported: {model}")
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
        if model not in available:
            raise ValueError(f"Model is not available to your API key: {model}")
        logger.success(f"Successfully validated model: {model}")

    async def initialize_topics(self) -> List[str]:
        """Ensure topics are initialized, i.e. topics already exist and are read,
        or a new list of topics is generated.
        """
        if os.path.exists(self.topics_path):
            self.topics = list(
                {
                    line.strip()
                    for line in open(self.topics_path).readlines()
                    if line.strip()
                }
            )
            logger.info(f"Using {len(self.topics)} topics from {self.topics_path}...")
            return

        logger.info("Generating random topics to use in prompts...")
        seen = set([])
        self.topics = []
        with open(self.topics_path, "w") as outfile:
            count = self.topic_request_count
            while count > 0:
                todo = 8 if count >= 8 else count
                responses = await asyncio.gather(
                    *[
                        self.generate_response(self.topic_prompt, **self.api_params)
                        for _ in range(todo)
                    ]
                )
                count -= todo
                for response in responses:
                    if not response:
                        continue
                    for topic in re.findall(
                        r"(?:^|\n)\d+\. (.*?)(?:$|(?=\n\d+\. ))", response, re.DOTALL
                    ):
                        if (
                            not topic
                            or topic.strip().endswith(":")
                            or topic.lower().strip() in seen
                        ):
                            continue
                        seen.add(topic.lower().strip())
                        self.topics.append(topic)
                        outfile.write(topic.strip() + "\n")
        logger.success(
            f"Successfully generated {len(self.topics)} topics, stored in {self.topics_path}..."
        )

    def get_instructor_topics(self, instructor_config):
        """Get the topics for a specific instructor, defaulting to main topics.

        :param instructor_config: Dict containing the target instructor's config.
        :type instructor_config: dict

        :return: List of topic strings.
        :rtype: list[str]
        """
        if not instructor_config.get("topics_path"):
            return self.topics
        with open(instructor_config["topics_path"]) as infile:
            topics = list({line.strip() for line in infile.readlines() if line.strip()})
            if not topics:
                raise ValueError(
                    f"Found empty topics file: {instructor_config['topics_path']}"
                )
        return topics

    @backoff.on_exception(
        backoff.fibo,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            ServerError,
            RateLimitError,
            TooManyRequestsError,
            ServerOverloadedError,
        ),
        max_value=19,
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
        request_id = str(uuid4())
        logger.debug(f"POST [{request_id}] with payload {json.dumps(payload)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENAI_API_BASE_URL}{path}",
                headers=headers,
                json=payload,
                timeout=600.0,
            ) as result:
                if result.status != 200:
                    text = await result.text()
                    logger.error(f"OpenAI request error: {text}")
                    if "too many requests" in text.lower():
                        raise TooManyRequestsError(text)
                    if "rate limit reached" in text.lower():
                        sleep(30)
                        raise RateLimitError(text)
                    elif "context_length_exceeded" in text.lower():
                        raise ContextLengthExceededError(text)
                    elif "server_error" in text and "overloaded" in text.lower():
                        raise ServerOverloadedError(text)
                    elif (
                        "bad gateway" in text.lower() or "server_error" in text.lower()
                    ):
                        raise ServerError(text)
                    else:
                        raise BadResponseError(text)
                result = await result.json()
                logger.debug(f"POST [{request_id}] response: {json.dumps(result)}")
                self.used_tokens += result["usage"]["total_tokens"]
                if self.max_tokens and self.used_tokens > self.max_tokens:
                    raise TokensExhaustedError(
                        f"Max token usage exceeded: {self.used_tokens}"
                    )
                logger.debug(f"token usage: {self.used_tokens}")
                return result

    async def _post_no_exc(self, *a, **k):
        """Post, ignoring all exceptions."""
        try:
            return await self._post(*a, **k)
        except Exception as ex:
            logger.error(f"Error performing post: {ex}")
        return None

    async def generate_response(self, instruction: str, **kwargs) -> str:
        """Call OpenAI with the specified instruction and return the text response.

        :param instruction: The instruction to respond to.
        :type instruction: str

        :return: Response text.
        :rtype: str
        """
        filter_response = kwargs.pop("filter_response", True)
        model = kwargs.get("model", self.model)
        completions = True if model in MODEL_ENDPOINTS["completions"] else False
        path = "/v1/completions" if completions else "/v1/chat/completions"
        payload = {**kwargs}
        if "model" not in payload:
            payload["model"] = model
        if completions:
            payload["prompt"] = instruction
            payload["max_tokens"] = 2000
        else:
            payload["messages"] = [{"role": "user", "content": instruction}]
        response = await self._post_no_exc(path, payload)
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

        if filter_response:
            for banned in self.response_filters:
                if banned.search(text, re.I):
                    logger.warning(f"Banned response [{banned}]: {text}")
                    return None
            if text.startswith(("I'm sorry,", "Apologies,", "I can't", "I won't")):
                logger.warning(f"Banned response [apology]: {text}")
                return None
        return text

    async def is_too_similar(self, instruction: str, min_score: float = None):
        """Check the similarity of a new instruction to the existing set.

        :param instruction: The instruction string to compare.
        :type instruction: str

        :param min_score: Minimum document similarity score to consider unique.
        :type min_score: float

        :return: Boolean indicating if the instruction is too similar or not.
        :rtype: bool
        """
        async with self.docstore_lock:
            min_ = 1.0
            for docstore in self.docstores:
                similar = docstore.similarity_search_with_score(instruction, k=1)
                for _, score in similar:
                    if score < min_:
                        min_ = score
            if min_ <= min_score:
                logger.warning(
                    f"Skipping instruction, too similar [{min_}]: {instruction}"
                )
                return True
            return False

    def persist(self, item):
        """Persist a single item to the output file and docstore."""
        self.outfile.write(json.dumps(item) + "\n")
        self.outfile.flush()
        self.docstores[-1].add_texts([item["instruction"]])
        self.docstore_size += 1
        if self.docstore_size >= MAX_DOCSTORE_SIZE:
            logger.info("Initializing new docstore...")
            self.docstores.append(
                Chroma.from_texts(["__initialize__"], self.embeddings)
            )
            self.docstore_size = 0

    async def run_instructor(self, category, method_map):
        """Run a single instructor, as an async task."""
        if category not in method_map:
            logger.warning(f"Unknown category: {category}, skipping...")
            return
        logger.info(f"Generating instructions for {category}...")
        started_at = datetime.datetime.now()
        running_total = self.instructor_counts.get(category, 0)
        async for item in method_map[category](self):
            self.persist(item)
            running_total += 1
            logger.success(
                f"Generated unique instruction [{category}, total={running_total}]: {item['instruction'][:100]}"
            )
        delta = (datetime.datetime.now() - started_at).total_seconds()
        logger.success(
            f"Finished generating {running_total} instructions [{category}] in {delta} seconds."
        )

    async def run(self):
        """Run prompt generation and answer to completion."""
        from airoboros.instructors.agent import generate as agent_generator
        from airoboros.instructors.card import generate as card_generator
        from airoboros.instructors.coding import generate as coding_generator
        from airoboros.instructors.contextual import generate as contextual_generator
        from airoboros.instructors.cot import generate as cot_generator
        from airoboros.instructors.counterfactual_contextual import (
            generate as counterfactual_contextual_generator,
        )
        from airoboros.instructors.experience import generate as experience_generator
        from airoboros.instructors.general import generate as general_generator
        from airoboros.instructors.orca import generate as orca_generator
        from airoboros.instructors.plan import generate as plan_generator
        from airoboros.instructors.riddle import generate as riddle_generator
        from airoboros.instructors.roleplay import generate as roleplay_generator
        from airoboros.instructors.trivia import generate as trivia_generator
        from airoboros.instructors.wordgame import generate as wordgame_generator
        from airoboros.instructors.writing import generate as writing_generator

        method_map = {
            "agent": agent_generator,
            "card": card_generator,
            "coding": coding_generator,
            "contextual": contextual_generator,
            "cot": cot_generator,
            "counterfactual_contextual": counterfactual_contextual_generator,
            "experience": experience_generator,
            "general": general_generator,
            "plan": plan_generator,
            "orca": orca_generator,
            "riddle": riddle_generator,
            "roleplay": roleplay_generator,
            "trivia": trivia_generator,
            "wordgame": wordgame_generator,
            "writing": writing_generator,
        }

        await self.initialize_topics()
        self.initialize_docstores()

        # Generate instructions for each category.
        self.outfile = open(self.output_path, "a+")
        started_at = datetime.datetime.now()
        try:
            tasks = [
                asyncio.create_task(self.run_instructor(category, method_map))
                for category in self.instructors
            ]
            for task in tasks:
                await task
        finally:
            self.outfile.close()
        delta = (datetime.datetime.now() - started_at).total_seconds()
        logger.success(
            f"Finished generating all instructions in {delta} seconds, enjoy!"
        )


def generate_instructions(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    asyncio.run(SelfInstructor(**vars(parser.parse_args(args))).run())


def generate_topics(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    instructor = SelfInstructor(**vars(parser.parse_args(args)))
    asyncio.run(instructor.initialize_topics())


if __name__ == "__main__":
    generate_instructions(sys.argv[1:])
