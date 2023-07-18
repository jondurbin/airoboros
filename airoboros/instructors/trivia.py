import re
from loguru import logger
from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor):
    """Generator for trivia training data."""
    async for item in generate_inline(instructor, "trivia"):
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
        yield item
