import asyncio
import glob
import json
import os
import random
from loguru import logger


async def generate(instructor):
    """Generator for detailed writing training data."""
    config = instructor.instructors.get("detailed_writing", {})
    if not config:
        return
    target_count = config.get("count")
    if target_count is None:
        target_count = instructor.default_count
    target_count = int(target_count)
    if not target_count:
        return

    # Load the seeds tasks.
    seed_path = config.get("seed_path", "detailed_writing_seeds")
    if not os.path.isdir(seed_path):
        seed_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prompts", seed_path
        )
        if not os.path.isdir(seed_path):
            raise Exception("No seeds!")
    seeds = []
    for path in glob.glob(os.path.join(seed_path, "*.txt")):
        with open(str(path)) as infile:
            seeds.append(infile.read())
    seed_index = 0

    # Load the prompt template.
    path = config.get("prompt_path", "detailed_writing.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        template = infile.read()

    # Load the response generating prompt template.
    path = config.get("response_prompt_path", "detailed_writing_response.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        response_template = infile.read()

    # Load the topics.
    topics = instructor.get_instructor_topics(config)
    random.shuffle(topics)
    topic_index = 0

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score")
    if min_score is None:
        min_score = instructor.min_docsearch_score
    min_score = float(min_score)

    # Generate the instruction/response pairs until we reach the target count.
    if "detailed_writing" not in instructor.instructor_counts:
        instructor.instructor_counts["detailed_writing"] = 0
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    futures = []
    language = config.get("language") or instructor.language
    flesch = config.get("flesch") or instructor.default_flesch
    while instructor.instructor_counts["detailed_writing"] < target_count:
        # Generate the prompts.
        topic = topics[topic_index]
        topic_index += 1
        if topic_index >= len(topics):
            topic_index = 0
        prompt = template.format(
            example=seeds[seed_index],
            flesch=flesch,
            language=language,
            topic=json.dumps(topic),
            topic_avoidance=instructor.topic_avoidance,
        )
        seed_index += 1
        if seed_index >= len(seeds):
            seed_index = 0
        futures.append(instructor.generate_response(prompt, **api_params))
        if len(futures) < batch_size:
            continue

        instructions = []
        for instruction in await asyncio.gather(*futures):
            if not instruction or not instruction.strip():
                continue
            if await instructor.is_too_similar(instruction, min_score=min_score):
                logger.warning("Skipping detailed writing prompt, too similar.")
                continue
            instructions.append(instruction)
        if not instructions:
            futures = []
            continue

        # Generate the responses.
        futures = [
            instructor.generate_response(
                response_template.format(
                    instruction=instruction, flesch=flesch, language=language
                ),
                **api_params,
            )
            for instruction in instructions
        ]
        responses = await asyncio.gather(*futures)
        for idx in range(len(responses)):
            response = responses[idx]
            if not response or not response.strip():
                continue
            yield {
                "instruction": instructions[idx].strip(),
                "response": response.strip(),
                "category": "detailed_writing",
            }
            if instructor.instructor_counts["detailed_writing"] >= target_count:
                break
        futures = []
