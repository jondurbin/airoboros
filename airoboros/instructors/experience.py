import asyncio
import os
import re


async def generate(instructor, **kwargs):
    """Generator for experiences."""
    config = instructor.instructors.get("experience")
    if not config:
        return
    target_count = config.get("count")
    if target_count is None:
        target_count = instructor.default_count
    target_count = int(target_count)

    # Load the prompt template.
    path = config.get("prompt_path", "experience.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        prompt = infile.read()

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score")
    if min_score is None:
        min_score = instructor.min_docsearch_score
    min_score = float(min_score)

    # Generate the instruction/response pairs until we reach the target count.
    if "experience" not in instructor.instructor_counts:
        instructor.instructor_counts["experience"] = 0
    language = config.get("language") or instructor.language
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    flesch = config.get("flesch") or instructor.default_flesch
    word_count = config.get("word_count") or 800
    futures = []
    while instructor.instructor_counts["experience"] < target_count:
        futures.append(
            instructor.generate_response(
                prompt.format(language=language, flesch=flesch, word_count=word_count),
                messages=kwargs.get("messages", []),
                filter_response=False,
                **api_params
            )
        )
        if len(futures) < batch_size:
            continue
        for response in await asyncio.gather(*futures):
            if not response or not response.strip():
                continue

            # Parse instructions and generate responses.
            match = re.search(
                r"SETTING:\s*(.*?)EXPERIENCE:\s*(.*)", response, re.DOTALL
            )
            if not match:
                continue

            instruction = match.group(1).strip()
            response = match.group(2).strip()
            if (
                not instruction
                or not response
                or await instructor.is_too_similar(instruction, min_score=min_score)
            ):
                continue
            yield {
                "instruction": instruction,
                "response": response,
                "category": "experience",
            }
            if instructor.instructor_counts["experience"] >= target_count:
                break
        futures = []
