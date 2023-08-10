import re
from loguru import logger
from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor, **kwargs):
    """Generator for trivia training data."""
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
        ] = "You are a world class trivia bot - generate accurate, succint responses."
        yield item
