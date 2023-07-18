import asyncio
import os
import re


async def generate(instructor, category):
    """Generator for simple instruction response tasks (e.g. roleplay, wordgames)."""
    config = instructor.instructors.get(category)
    if not config:
        return
    target_count = config.get("count") or instructor.default_count

    # Load the prompt template.
    path = config.get("prompt_path", f"{category}.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        template = infile.read()

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score") or instructor.min_docsearch_score

    # Generate the instruction/response pairs until we reach the target count.
    batch_size = config.get("batch_size") or instructor.default_batch_size
    count = instructor.instructor_counts.get(category, 0)
    language = config.get("language") or instructor.language
    while count < target_count:
        # Get a batch of instructions.
        prompt = (
            template.format(batch_size=batch_size, language=language)
            if "{topic_avoidance}" not in template
            else template.format(
                batch_size=batch_size,
                language=language,
                topic_avoidance=instructor.topic_avoidance,
            )
        )
        response = await instructor.generate_response(prompt, **api_params)
        if not response:
            continue

        # Parse instructions and generate responses.
        futures = []
        instructions = []
        for instruction in re.findall(
            r"(?:^|\n)TSK \d+\. (.*?)(?:$|(?=\nTSK \d+\. ))", response, re.DOTALL
        ):
            if not instruction.strip() or await instructor.is_too_similar(
                instruction, min_score=min_score
            ):
                continue
            instructions.append(instruction)
            futures.append(instructor.generate_response(instruction, **api_params))
        if not futures:
            continue
        responses = await asyncio.gather(*futures)
        for idx in range(len(responses)):
            response = responses[idx]
            if not response or not response.strip():
                continue
            yield {
                "instruction": instructions[idx].strip(),
                "response": response.strip(),
                "category": category,
            }
            count += 1
            if count >= target_count:
                break
