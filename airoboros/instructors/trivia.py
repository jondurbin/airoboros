import asyncio
import random
import re
from loguru import logger
from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor, **kwargs):
    """Generator for trivia training data."""
    config = instructor.instructors.get("trivia")
    if not config:
        return
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    batch = []
    async for item in generate_inline(instructor, "trivia", **kwargs):
        # Double check some wordgame style questions.
        match = re.search(
            r"(start|end)(?:s|ing) with ['\"]([^'\"]+)['\"]", item["instruction"]
        )
        if match:
            if (
                match.group(1) == "start"
                and not item["response"].lower().startswith(match.group(2).lower())
            ) or (
                match.group(1) == "end"
                and not item["response"].lower().endswith(match.group(2).lower())
            ):
                logger.warning(
                    f"Validation failure: {item['instruction']} -- {item['response']}"
                )
                continue
        match = re.search(
            r"(?:with|contain(?:s|ing)|hav(?:e|ing)) ['\"]([^'\"]+)['\"]",
            item["instruction"],
        )
        if match and match.group(1) not in item["response"]:
            logger.warning(
                f"Validation failure: {item['instruction']} -- {item['response']}"
            )
            continue
        item[
            "system"
        ] = "You are a world class trivia AI - provide accurate, succint responses."
        yield item

        # We also want to generate the non-trivia version of the responses, to ensure our system
        # prompts are respected properly, i.e. short responses if trivia bot, otherwise standard.
        if random.random() < 0.5:
            batch.append(
                item["instruction"]
                + "  Include a brief bit of detail/context in your response, but don't repeat/restate the input."
            )
        if len(batch) < batch_size:
            continue
        responses = await asyncio.gather(
            *[
                instructor.generate_response(batch[idx], **api_params)
                for idx in range(len(batch))
            ]
        )
        for idx in range(len(batch)):
            if not responses[idx] or not responses[idx].strip():
                continue
            yield {
                "category": "trivia",
                "instruction": batch[idx],
                "response": responses[idx],
            }
        batch = []
