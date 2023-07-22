import os
import re


async def generate(
    instructor, category, start_key="QUESTION", end_key="ANSWER", filter_response=True
):
    """Generator for generic inline question answer pair training data."""
    config = instructor.instructors.get(category)
    if not config:
        return
    target_count = config.get("count")
    if target_count is None:
        target_count = instructor.default_count
    target_count = int(target_count)
    if not target_count:
        return

    # Load the prompt template.
    path = config.get("prompt_path", f"{category}.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        template = infile.read()

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score")
    if min_score is None:
        min_score = instructor.min_docsearch_score
    min_score = float(min_score)

    # Generate the instruction/response pairs until we reach the target count.
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    count = instructor.instructor_counts.get(category, 0)
    language = config.get("language") or instructor.language
    while count < target_count:
        # Get a batch of instructions.
        prompt = (
            template.format(batch_size=batch_size, language=language)
            if "{batch_size}" in template
            else template.format(language=language)
        )
        response = await instructor.generate_response(
            prompt, filter_response=filter_response, **api_params
        )
        if not response:
            continue

        # Parse instructions and generate responses.
        for instruction, response in re.findall(
            f"{start_key}:(.*?){end_key}:(.*?)(?={start_key}|$)", response, re.DOTALL
        ):
            if not instruction.strip() or await instructor.is_too_similar(
                instruction, min_score=min_score
            ):
                continue
            yield {
                "instruction": instruction.strip(),
                "response": response.strip(),
                "category": category,
            }
            count += 1
            if count >= target_count:
                break
