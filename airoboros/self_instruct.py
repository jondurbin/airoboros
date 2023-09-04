import aiohttp
import argparse
import asyncio
import backoff
import copy
import datetime
import faiss
import os
import json
import math
import numpy as np
import random
import re
import requests
import secrets
import sys
import yaml
from collections import defaultdict
from loguru import logger
from time import sleep
from tqdm import tqdm
from typing import List, Dict, Any
from uuid import uuid4
from airoboros.embeddings import calculate_embeddings
from airoboros.exceptions import (
    RateLimitError,
    TooManyRequestsError,
    TokensExhaustedError,
    ServerOverloadedError,
    ServerError,
    ContextLengthExceededError,
    BadResponseError,
)
from fast_sentence_transformers import FastSentenceTransformer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Defaults and constants.
MAX_DOCSTORE_SIZE = 15000
OPENAI_API_BASE_URL = "https://api.openai.com"
READABILITY_HINT = "The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  Only output the story - don't add any notes or information about Flesch-Kincaid scores."


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
        "--debug": {
            "action": "store_true",
            "help": "enable debug logging",
        },
    }

    def __init__(self, *, config_path: str = "config.yaml", debug: bool = False):
        """Constructor."""
        if not debug:
            logger.remove()
            logger.add(sys.stdout, level="INFO")
        self.used_tokens = 0
        self.config_path = config_path
        self.load_config()
        self.instructor_counts = defaultdict(int)

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
            "presence_penalty": float(api_params.get("presence_penalty") or 0.0),
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
        self.default_flesch = raw_config.get("default_flesch") or READABILITY_HINT

        # Embedding model.
        model_name = raw_config.get("embedding_model") or "thenlper/gte-small"

        # Hacky, but we'll load this twice, the first time to get dimension, since
        # it's not accessible in the Fast (cpu) version.
        model = SentenceTransformer(model_name)
        self.embedding_dimension = model.get_sentence_embedding_dimension()
        model = None
        if raw_config.get("embedding_device") == "cuda":
            self.embedding_model = SentenceTransformer(model_name, device="cuda")
        else:
            self.embedding_model = FastSentenceTransformer(model_name, device="cpu")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.index = faiss.IndexFlatL2(self.embedding_dimension)

        # Validate the model for each generator.
        self.instructors = raw_config.get("instructors")
        self.validate_model(self.model)
        valid_models = {self.model: True}
        for key, config in self.instructors.items():
            if config.get("model") and config["model"] not in valid_models:
                self.validate_model(config["model"])
                valid_models[config["model"]] = True

    def initialize_index(self):
        """Initialize the in-memory faiss index to check prompt uniqueness."""
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
                        category = task.get("category", "general")
                        if category != "rp" or "rp" in task:
                            self.instructor_counts[category] += 1
                        if task["category"] != "rp":
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
        logger.info("Initializing faiss index similarity comparison...")
        if not docs:
            docs = ["__initialize__"]

        # This is a bit slow.
        for doc in tqdm(docs):
            self.index.add(
                np.array(
                    [
                        calculate_embeddings(
                            doc, self.embedding_model, self.embedding_tokenizer
                        )
                    ]
                )
            )

    def validate_model(self, model):
        """Ensure the specified model is available."""
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

    @staticmethod
    def load_template(path: str) -> str:
        """Load a prompt template."""
        if not os.path.exists(path):
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "instructors",
                "prompts",
                path,
            )
        with open(path) as infile:
            return infile.read()

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
                    if (
                        "rate limit reached" in text.lower()
                        or "rate_limit_exceeded" in text.lower()
                    ):
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
        messages = copy.deepcopy(kwargs.pop("messages", None) or [])
        filter_response = kwargs.pop("filter_response", True)
        model = kwargs.get("model", self.model)
        path = "/v1/chat/completions"
        payload = {**kwargs}
        if "model" not in payload:
            payload["model"] = model
        payload["messages"] = messages
        if instruction:
            payload["messages"].append({"role": "user", "content": instruction})
        response = await self._post_no_exc(path, payload)
        if (
            not response
            or not response.get("choices")
            or response["choices"][0]["finish_reason"] == "length"
        ):
            return None
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

    async def is_decent_response(self, item):
        """Filter the responses by having the LLM score based on a set of rules."""
        config = self.raw_config.get("scoring", {})
        template = self.load_template(config.get("prompt_path") or "filter.txt")
        api_params = {**self.api_params, **config.get("api_params", {})}
        instruction = item["instruction"]
        if item.get("category") == "coding" and "PLAINFORMAT" in instruction:
            instruction = instruction.replace("PLAINFORMAT", "").strip()
            instruction += "\n" + "\n".join(
                [
                    "Generate only the code, as a single, plain text output.",
                    "Do not include an intro sentence indicating what the code will do.",
                    "Do not include any instructions for usage, warnings about replacing certain values, etc.",
                    "Do not surround the code with backticks/markdown formatting.",
                    "Include help code comments.",
                ]
            )
        system_prompt = ""
        if item.get("system"):
            system_prompt = "\n".join(
                [
                    "- did the response respect the system prompt that was used?",
                    "SYSTEM PROMPT:",
                    item["system"],
                ]
            )
        result = await self.generate_response(
            template.format(
                instruction=item["instruction"],
                response=item["response"],
                threshold=config.get("threshold") or "100",
                system_prompt=system_prompt,
                filter_response=False,
            ),
            **api_params,
        )
        preview = item["instruction"].splitlines()[0][0:100]
        if len(preview) == 100:
            preview += "..."
        if not result:
            logger.error(
                f"Error evaluating response, assuming decent [{item['category']}]: {preview}"
            )
            return True
        if "GOOD" in result:
            logger.info(f"Judge: good [{item['category']}]: {preview}")
            return True
        logger.info(f"Judge: bad [{item['category']}]: {preview}")
        return False

    async def judge(self, instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter only the "good" instructions, as determined by an LLM."""
        batch_size = (
            self.raw_config.get("judge", {}).get("batch_size")
            or self.default_batch_size
        )
        batches = np.array_split(
            instructions, math.ceil(len(instructions) / batch_size)
        )
        quality = []
        for batch in batches:
            results = await asyncio.gather(
                *[self.is_decent_response(item["item"]) for item in batch]
            )
            for idx in range(len(batch)):
                if results[idx]:
                    quality.append(batch[idx])
        return quality

    async def cull(self, input_paths: List[str], output_path: str) -> None:
        """Use the LLM to filter bad responses based on a set of rules.

        :param input_paths: List of paths to the input JSONL file(s) to filter.
        :type input_paths: List[str]

        :param output_path: Path to save the "good" instructions to.
        :type output_path: str

        """
        # See if we have any state data.
        state = {}
        if os.path.exists(f"{output_path}.state"):
            with open(f"{output_path}.state") as infile:
                state = json.loads(infile.read())
                logger.info(
                    f"Resuming from previous cull state - to restart, delete `{output_path}.state`"
                )

        def _save_state(c, line):
            nonlocal state
            state[c] = line
            with open(f"{output_path}.state", "w") as outfile:
                outfile.write(json.dumps(state, indent=2) + "\n")

        categories = defaultdict(list)
        found = set([])
        for path in input_paths:
            with open(path) as infile:
                for line in infile.readlines():
                    item = json.loads(line)
                    category = item.get("category", "general")
                    if category == "reasoning_or_math":
                        category = "orca"
                        item["category"] = category

                    # Skip items already processed.
                    if category in state:
                        if line == state[category]:
                            found.add(category)
                            continue
                        elif category not in found:
                            continue
                    categories[category].append({"item": item, "line": line})

        # Deduplicate and select best items.
        output_file = open(output_path, "a+")
        max_k = self.raw_config.get("cull_max_k")
        if max_k is None:
            max_k = 1000
        for category in sorted(list(categories)):
            items = categories[category]
            # Skip categories that are too weird/cumbersome to score properly.
            if category in [
                "stylized_response",
                "rp",
                "detailed_writing",
                "contextual",
                "counterfactual_contextual",
                "plan",
                "song",
                "wordgame",
            ]:
                for idx in range(len(items)):
                    item = items[idx]["item"]
                    output_file.write(json.dumps(item) + "\n")
                    _save_state(category, items[idx]["line"])
                output_file.flush()
                continue

            # Add all of the items in this category to a faiss index.
            logger.info(
                f"Initializing faiss index for {category} with {len(items)} documents..."
            )
            index = faiss.IndexFlatL2(self.embedding_dimension)
            all_embeddings = []
            for item in tqdm(items):
                all_embeddings.append(
                    np.array(
                        [
                            calculate_embeddings(
                                "\n".join(
                                    [
                                        item["item"]["instruction"],
                                        item["item"]["response"],
                                    ]
                                ),
                                self.embedding_model,
                                self.embedding_tokenizer,
                            )
                        ]
                    )
                )
                index.add(all_embeddings[-1])

            # Here's where it's tricky...
            #
            # We need to iterate through the objects, finding all matches that are under are
            # specified similarity score for this category.
            #
            # Once we've found all of the matches, we can select the "best" by first using
            # the LLM to judge whether the response is high quality or not, but only if it's
            # a category that we can score well.
            #
            # If multiple instructions remain that are high quality, we can use other metrics,
            # such as length and complexity of speech to select the best.
            #
            # If none of the matching instructions are high quality, I guess we just remove
            # all of them?
            purged = set([])
            saved = set([])
            min_score = (
                self.instructors.get(category, {}).get("min_docsearch_score")
                or self.min_docsearch_score
            )
            for idx in range(len(items)):
                if idx in purged or idx in saved:
                    continue
                distances, indices = index.search(
                    all_embeddings[idx], k=min(len(items), max_k)
                )
                distances = distances[0].tolist()[1:]
                indices = indices[0].tolist()[1:]
                batch = [items[idx]]
                batch_idx = [idx]
                for check_idx in range(len(distances)):
                    # Don't check items before this one (since they would have already been checked).
                    if indices[check_idx] < idx:
                        continue

                    # Don't check items we've judged as duplicate or low-quality.
                    if indices[check_idx] in purged:
                        continue

                    # Ignore coding instructions that don't match on PLAINFORMAT tag.
                    if category == "coding":
                        source_has_plain = (
                            "PLAINFORMAT" in items[idx]["item"]["instruction"]
                        )
                        target_has_plain = (
                            "PLAINFORMAT"
                            in items[indices[check_idx]]["item"]["instruction"]
                        )
                        if (source_has_plain and not target_has_plain) or (
                            target_has_plain and not source_has_plain
                        ):
                            continue

                    # Ignore and stop checking if we exceed the min score.
                    if distances[check_idx] > min_score:
                        break
                    batch.append(items[indices[check_idx]])
                    batch_idx.append(indices[check_idx])
                # Score the batch.
                quality = await self.judge(batch)
                if not quality:
                    for purge_idx in range(len(batch)):
                        purged.add(batch_idx[purge_idx])
                        preview = items[batch_idx[purge_idx]]["item"][
                            "instruction"
                        ].splitlines()[0][0:100]
                        logger.warning(f"Removing low-quality instruction: {preview}")
                    continue

                # Only one high-quality result, keep it.
                if len(quality) == 1:
                    preview = quality[0]["item"]["instruction"].splitlines()[0][0:100]
                    logger.success(f"Saving high-quality instruction: {preview}")
                    output_file.write(json.dumps(quality[0]["item"]) + "\n")
                    output_file.flush()
                    _save_state(category, quality[0]["line"])
                    found = False
                    for save_idx in range(len(batch)):
                        if batch[save_idx] == quality[0]:
                            if not found:
                                saved.add(batch_idx[save_idx])
                                found = True
                            else:
                                purged.add(batch_idx[save_idx])
                    continue

                # This is kind of a hacky fallback, but it's fast and easy.
                longest = sorted(
                    quality,
                    key=lambda x: len(x["item"]["instruction"] + x["item"]["response"]),
                )[-1]
                found = False
                for purge_idx in range(len(batch)):
                    if batch[purge_idx] == longest and not found:
                        found = True
                        saved.add(batch_idx[purge_idx])
                    if batch[purge_idx] != longest or found:
                        purged.add(batch_idx[purge_idx])
                        found = True
                preview = longest["item"]["instruction"].splitlines()[0][0:100]
                logger.success(f"Saving high-quality, longest instruction: {preview}")
                output_file.write(json.dumps(longest) + "\n")
                output_file.flush()
                _save_state(category, longest["line"])
        output_file.close()

    async def is_too_similar(
        self, input_text: str, min_score: float = None, index=None
    ):
        """Check the similarity of an input string against an index.

        :param input_text: The input string to calculate similarity of.
        :type input_text: str

        :param min_score: Minimum document similarity score to consider unique.
        :type min_score: float

        :param index: Optional faiss index to query against, defaults to main index.
        :type index: failss index

        :return: Boolean indicating if the text is too similar or not.
        :rtype: bool
        """
        index = index or self.index
        input_embeddings = np.array(
            [
                calculate_embeddings(
                    input_text, self.embedding_model, self.embedding_tokenizer
                )
            ]
        )
        min_score = min_score or self.min_docsearch_score
        distance, _ = index.search(input_embeddings, k=1)
        distance = distance[0].tolist()
        if not distance:
            return False
        if distance[0] <= min_score:
            logger.warning(f"Too similar [{distance[0]}]: {input_text}")
            return True
        return False

    def persist(self, item):
        """Persist a single item to the output file and add it to the index."""
        skip_counting = item.pop("skip_counting", False)
        if "instruction" in item:
            item["instruction"] = item["instruction"].strip()
        if "response" in item:
            item["response"] = item["response"].strip()
        if "system" in item:
            item["system"] = item["system"].strip()
        self.outfile.write(json.dumps(item) + "\n")
        self.outfile.flush()
        if item["category"] != "rp":
            self.index.add(
                np.array(
                    [
                        calculate_embeddings(
                            item["instruction"],
                            self.embedding_model,
                            self.embedding_tokenizer,
                        )
                    ]
                )
            )
        if not skip_counting:
            self.instructor_counts[item["category"]] += 1

    async def run_instructor(self, category, method_map, **kwargs):
        """Run a single instructor, as an async task."""
        if category not in method_map:
            logger.warning(f"Unknown category: {category}, skipping...")
            return
        logger.info(f"Generating instructions for {category}...")
        started_at = datetime.datetime.now()
        running_total = self.instructor_counts.get(category, 0)
        async for item in method_map[category](self, **kwargs):
            self.persist(item)
            preview = None
            if category == "rp":
                if "rp" in item:
                    running_total += 1
                    preview = item["rp"][0]["content"].splitlines()[0][:100]
            else:
                running_total += 1
                preview = item["instruction"].splitlines()[0][0:100]
            if preview:
                logger.success(
                    f"Generated unique instruction [{category}, total={running_total}]: {preview}"
                )
        delta = (datetime.datetime.now() - started_at).total_seconds()
        logger.success(
            f"Finished generating {running_total} instructions [{category}] in {delta} seconds."
        )

    async def run(self):
        """Run prompt generation and answer to completion."""
        from airoboros.instructors.agent import generate as agent_generator
        from airoboros.instructors.awareness import generate as awareness_generator
        from airoboros.instructors.card import generate as card_generator
        from airoboros.instructors.coding import generate as coding_generator
        from airoboros.instructors.contextual import generate as contextual_generator
        from airoboros.instructors.cot import generate as cot_generator
        from airoboros.instructors.counterfactual_contextual import (
            generate as counterfactual_contextual_generator,
        )
        from airoboros.instructors.detailed_writing import (
            generate as detailed_writing_generator,
        )
        from airoboros.instructors.editor import generate as editor_generator
        from airoboros.instructors.experience import generate as experience_generator
        from airoboros.instructors.general import generate as general_generator
        from airoboros.instructors.gtkm import generate as gtkm_generator
        from airoboros.instructors.joke import generate as joke_generator
        from airoboros.instructors.misconception import (
            generate as misconception_generator,
        )
        from airoboros.instructors.multiple_choice import (
            generate as multiple_choice_generator,
        )
        from airoboros.instructors.orca import generate as orca_generator
        from airoboros.instructors.plan import generate as plan_generator
        from airoboros.instructors.riddle import generate as riddle_generator
        from airoboros.instructors.roleplay import generate as roleplay_generator
        from airoboros.instructors.rp import generate as rp_generator
        from airoboros.instructors.rp import generate_cards
        from airoboros.instructors.song import generate as song_generator
        from airoboros.instructors.stylized_response import (
            generate as stylized_response_generator,
        )
        from airoboros.instructors.trivia import generate as trivia_generator
        from airoboros.instructors.wordgame import generate as wordgame_generator
        from airoboros.instructors.writing import generate as writing_generator

        method_map = {
            "agent": agent_generator,
            "awareness": awareness_generator,
            "card": card_generator,
            "coding": coding_generator,
            "contextual": contextual_generator,
            "cot": cot_generator,
            "counterfactual_contextual": counterfactual_contextual_generator,
            "detailed_writing": detailed_writing_generator,
            "experience": experience_generator,
            "general": general_generator,
            "joke": joke_generator,
            "misconception": misconception_generator,
            "multiple_choice": multiple_choice_generator,
            "plan": plan_generator,
            "orca": orca_generator,
            "riddle": riddle_generator,
            "roleplay": roleplay_generator,
            "rp": rp_generator,
            "song": song_generator,
            "trivia": trivia_generator,
            "wordgame": wordgame_generator,
            "writing": writing_generator,
        }

        await self.initialize_topics()
        self.initialize_index()

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

        # Editor needs the writing data to run.
        if self.instructor_counts.get("writing"):
            logger.info("Generating editor prompts using existing writing data...")
            method_map["editor"] = editor_generator
            self.outfile = open(self.output_path, "a+")
            try:
                await self.run_instructor("editor", method_map)
            finally:
                self.outfile.close()

        # After we have a sampling of instructions, we can also generate a list of responses
        # based on character cards generated.
        if await generate_cards(self):
            logger.info(
                "Re-generating a sampling of responses using character cards..."
            )
            with open(self.output_path) as infile:
                existing = [json.loads(line) for line in infile.readlines()]
            method_map["stylized_response"] = stylized_response_generator
            method_map["gtkm"] = gtkm_generator
            self.outfile = open(self.output_path, "a+")
            tasks = []
            try:
                tasks.append(
                    asyncio.create_task(
                        self.run_instructor(
                            "stylized_response",
                            method_map,
                            existing=existing,
                        )
                    )
                )
                tasks.append(
                    asyncio.create_task(self.run_instructor("gtkm", method_map))
                )
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


def cull_instructions(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    parser.add_argument(
        "--input",
        **{
            "type": str,
            "help": "path to the file containing instructions to cull",
            "nargs": "+",
        },
    )
    parser.add_argument(
        "--output",
        **{
            "type": str,
            "default": "culled.jsonl",
            "help": "path to save the culled instructions to",
        },
    )
    all_args = vars(parser.parse_args(args))
    instructor = SelfInstructor(config_path=all_args["config_path"])
    asyncio.run(instructor.cull(all_args["input"], all_args["output"]))


if __name__ == "__main__":
    generate_instructions(sys.argv[1:])
