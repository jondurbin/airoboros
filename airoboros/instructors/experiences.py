import os
import re


async def generate(instructor):
    """Generator for experiences."""
    config = instructor.instructors.get("experiences")
    if not config:
        return
    target_count = config.get("count") or instructor.default_count

    # Load the prompt template.
    path = config.get("prompt_path", "experiences.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        prompt = infile.read()

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score") or instructor.min_docsearch_score

    # Generate the instruction/response pairs until we reach the target count.
    count = instructor.instructor_counts.get("experiences", 0)
    while count < target_count:
        # Get a batch of instructions.
        response = await instructor.generate_response(prompt, **api_params)
        if not response:
            continue

        # Parse instructions and generate responses.
        match = re.search(r"SETTING:\s*(.*?)EXPERIENCE:\s*(.*)", response, re.DOTALL)
        if not match:
            continue
        instruction = match.group(1).strip()
        response = match.group(2).strip()
        if (
            not instruction
            or not response
            or instructor.is_too_similar(instruction, min_score=min_score)
        ):
            continue
        yield {
            "instruction": instruction,
            "response": response,
            "category": "experiences",
        }
        count += 1
        if count >= target_count:
            break
