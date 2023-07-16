import json
import os
import re


async def generate(instructor):
    """Generator for wordgame training data."""
    config = instructor.instructors.get("wordgames")
    if not config:
        return
    target_count = config.get("count", instructor.default_count)

    # Load the prompt template.
    path = config.get("prompt_path", "wordgames.txt")
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prompts", path
        )
    with open(path) as infile:
        template = infile.read()

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score") or instructor.min_docsearch_score

    # Generate the instruction/response pairs until we reach the target count.
    batch_size = config.get("batch_size", 10)
    count = instructor.instructor_counts.get("wordgames", 0)
    while count < target_count:
        # Get a batch of instructions.
        prompt = template.format(batch_size=batch_size)
        response = await instructor.generate_response(prompt, **api_params)
        if not response:
            continue

        # Parse instructions and generate responses.
        for instruction in re.findall(
            r"(?:^|\n)TASK \d+\. (.*?)(?:$|(?=\nTASK \d+\. ))", response, re.DOTALL
        ):
            if not instruction.strip() or instructor.is_too_similar(
                instruction, min_score=min_score
            ):
                continue
            response = await instructor.generate_response(instruction, **api_params)
            if not response:
                continue
            yield {
                "instruction": instruction.strip(),
                "response": response.strip(),
                "category": "general",
            }
            count += 1
            if count >= target_count:
                break
