import asyncio
import os
import re
import random


async def generate(instructor):
    """Generator for coding training data."""
    config = instructor.instructors.get("coding")
    if not config:
        return
    target_count = config.get("count") or instructor.default_count

    # Load the prompt template.
    path = config.get("prompt_path", "coding.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        template = infile.read()

    # Additional configuration related to coding tasks.
    related = config.get("related_software", [])
    coding_languages = config.get("coding_languages", [])
    if not coding_languages:
        raise ValueError("At least one coding language must be configured.")

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score") or instructor.min_docsearch_score

    # Generate the instruction/response pairs until we reach the target count.
    batch_size = config.get("batch_size", 10)
    count = instructor.instructor_counts.get("coding", 0)
    language_index = 0
    language = config.get("language") or instructor.language
    while count < target_count:
        # Inject languages to use for this batch.
        current_languages = []
        for _ in range(batch_size):
            current_languages.append(coding_languages[language_index])
            language_index += 1
            if language_index >= len(coding_languages):
                language_index = 0
        languages_str = "\n".join(
            [
                f" * task {idx + 1} should ask the user to use {language}"
                for idx, language in enumerate(current_languages)
            ]
        )
        related_str = ""
        if batch_size > 3:
            related_str = f"One of the tasks should require interacting with {random.choice(related)}."

        # Get a batch of instructions.
        prompt = template.format(
            batch_size=batch_size,
            languages=languages_str,
            related_software=related_str,
            language=language,
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

            # Optionally add plain formatting.
            plain = False
            if random.random() < 0.5:
                plain = True

            full_instruction = (
                instruction
                if not plain
                else "\n".join(
                    [
                        instruction,
                        "  ".join(
                            [
                                "Generate only the code, as a single, plain text output.",
                                "Do not include an intro sentence indicating what the code will do.",
                                "Do not include any instructions for usage, warnings about replacing certain values, etc.",
                                "Do not surround the code with backticks/markdown formatting.",
                            ]
                        ),
                    ]
                )
            )
            instructions.append(
                instruction if not plain else instruction + " PLAINFORMAT"
            )
            futures.append(instructor.generate_response(full_instruction, **api_params))
        if not futures:
            continue
        responses = await asyncio.gather(*futures)
        for idx in range(len(futures)):
            response = responses[idx]
            if not response:
                continue
            if "PLAINFORMAT" in instructions[idx] and "```" in response:
                response = re.split(r"```[^\n]*(?:$|[\r\n])", response)[1].strip()
                if not response:
                    continue
            yield {
                "instruction": instructions[idx].strip(),
                "response": response.strip(),
                "category": "coding",
            }
            count += 1
            if count >= target_count:
                break
