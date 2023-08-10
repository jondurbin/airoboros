import asyncio
import glob
import json
import os
import random
from loguru import logger

DEFAULT_CATEGORIES = [
    "cot",
    "detailed_writing",
    "experience",
    "general",
    "joke",
    "orca",
    "riddle",
    "trivia",
    "wordgame",
]
RULES = """
Avoid any and all warnings, disclaimers, reminders, etc. that this is only roleplay.
If you absolutely cannot avoid adding some type of warning/disclaimer/reminder/etc., it must be at the end of the output, prefixed with "REMINDER:"
Don't start your response with "Certainly!", "Sure, " or other similar phrases, just output the response as the specified character.
"""


async def generate(instructor, existing=[], **kwargs):
    """Generator for stylized response training data."""

    # Load the various configs/settings.
    category = "stylized_response"
    conf = instructor.instructors.get(category, {})
    if not conf:
        return
    count = conf.get("count")
    if count is None:
        count = instructor.default_count
    if not count:
        return
    categories = conf.get("categories", DEFAULT_CATEGORIES)
    if not categories:
        return
    card_config = instructor.instructors.get("chat_card", {})
    if not card_config:
        return
    batch_size = conf.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    if category not in instructor.instructor_counts:
        instructor.instructor_counts[category] = 0

    # Load the existing character cards.
    cards = []
    cards_dir = card_config.get("output_dir", "chat_cards")
    if not os.path.isdir(cards_dir):
        os.makedirs(cards_dir, exist_ok=True)
    else:
        for path in glob.glob(os.path.join(cards_dir, "*.json")):
            with open(str(path)) as infile:
                cards.append(json.loads(infile.read()))
    if not cards:
        logger.warning("No cards found!")
        return

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **conf.get("api_params", {})}

    # Extract the source pool of instructions so we can select a random sampling.
    instructions = {}
    for category in categories:
        instructions[category] = []
    for instruction in existing:
        if instruction.get("category") not in instructions:
            continue
        instructions[instruction["category"]].append(instruction)

    # Re-generate responses using the style cards.
    card_index = random.randint(0, len(cards) - 1)
    futures = []
    batch_instructions = []
    batch_system_prompts = []
    batch_names = []
    while instructor.instructor_counts["stylized_response"] < count:
        if instructions:
            category = random.choice(list(instructions))
            if not instructions[category]:
                instructions.pop(category)
                continue
            system_prompt = "\n".join(
                [
                    f"You are to take on the role of: {cards[card_index]['name']}",
                    cards[card_index]["description"],
                    cards[card_index]["stay_in_character"],
                    RULES,
                ]
            )
            batch_names.append(cards[card_index]["name"])
            card_index += 1
            if card_index >= len(cards):
                card_index = 0
            batch_system_prompts.append(system_prompt)
            idx = random.randint(0, len(instructions[category]) - 1)
            instruction = instructions[category].pop(idx)["instruction"]
            batch_instructions.append(instruction)
            futures.append(
                instructor.generate_response(
                    instruction,
                    messages=[{"role": "system", "content": system_prompt}],
                    **api_params,
                )
            )
        if instructions and len(futures) < batch_size:
            continue
        responses = await asyncio.gather(*futures)
        for idx in range(len(futures)):
            logger.success(f"Generated stylized response as: {batch_names[idx]}")
            if not responses[idx] or not responses[idx].strip():
                continue
            yield {
                "category": "stylized_response",
                "system": batch_system_prompts[idx],
                "instruction": batch_instructions[idx],
                "response": responses[idx].strip(),
            }
        futures = []
        batch_names = []
        batch_instructions = []
        batch_system_prompts = []
