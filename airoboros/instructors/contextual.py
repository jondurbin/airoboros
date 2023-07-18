import asyncio
import json
import os
import random
import re
from loguru import logger


TARGET_OPTIONS = [
    "the first generated text block.",
    "the last generated text block.",
    "all generated text blocks.",
    "one or more of the generated text blocks.",
    "one, randomly selected generated text block.",
]
ASK_FOR_SOURCE = """
Instruct or ask the user to provide a source/references for the information, e.g. "What is your source?", "Please provide references.", "Cite.", "Source?", "[insert references]", "Add a citation.", or other similar variations.
"""
TASK_DISPLAY_OPTIONS = [
    "as a paragraph",
    "as multiple lines, one task per line",
    "as a bullet list",
]
TASK_DISPLAY = "The set of task(s) should be displayed as {task_format}."
REQUEST_FORMATTING = "One task should ask for output to be formatted in a specific way, such as {sample_formats}, or similar type of formatting that would be appropriate for the task."
VALID_FORMAT = re.compile(
    r"^[\s\n]*(?:BEGININPUT[\s\n]+BEGINCONTEXT(?:.*?)(?=ENDCONTEXT)ENDCONTEXT(?:.*?)(?=ENDINPUT)ENDINPUT[\s\n]*)+BEGININSTRUCTION.*ENDINSTRUCTION[\s\r\n]*$",
    re.DOTALL,
)


def generate_prompt(instructor, config, template, topic_iter):
    """Generate a prompt with random selection of template values."""

    # Number of input context blocks to generate.
    input_count = random.choice([1, 1, 1, 2, 2, 3, 4])

    # Configure the number of metadata key/value pairs per input block.
    metadata_counts = "\n".join(
        [
            f"- context block {idx + 1} should have {random.randint(0, 8)} metadata key/value pairs"
            for idx in range(input_count)
        ]
    )

    # Select which input context block to target with question(s).
    target_selection = random.choice(TARGET_OPTIONS)

    # Number of questions/instructions to include for the given prompt.
    task_count = random.choice([1, 1, 1, 2, 2, 3])

    # Select the writing style for each context block.
    styles = ""
    if config.get("context_styles"):
        styles = "\n".join(
            [
                f"- text {idx + 1} should be in the form of: {random.choice(config['context_styles'])}"
                for idx in range(input_count)
            ]
        )

    # Ask for specific output format in one of the tasks.
    format_task = ""
    if config.get("formatting_options") and random.random() <= 0.2:
        sample = random.sample(
            config["formatting_options"], min(len(config["formatting_options"]), 3)
        )
        format_task = REQUEST_FORMATTING.format(sample_formats=", ".join(sample))

    # Add instructions indicating how context blocks should relate to each other.
    reference_texts = "The random text blocks should not reference each other."
    if random.random() <= 0.1 and input_count > 1:
        reference_texts = (
            "One of the random text blocks should reference details in another"
        )
        if random.random() <= 0.5:
            reference_texts += " using metadata, e.g. a link from one text block should be referenced in the text of another."
        else:
            reference_texts += "."

    # Add instructions to include source/reference information.
    include_source = ""
    if random.random() <= 0.2:
        include_source = ASK_FOR_SOURCE
        if config.get("formatting_options") and random.random() <= 0.2:
            include_source += f" Ask for the references/source as: {random.choice(config['formatting_options'])}"

    # Add instructions to add some confounders, i.e. unanswerable questions.
    task_confounder = ""
    if random.random() < 0.1:
        if task_count > 1:
            if random.random() < 0.5:
                task_confounder = "One of the tasks should be completely unrelated to the generated text(s)."
            else:
                task_confounder = "One of the tasks should be somewhat related to provided input text(s), but not answerable by any of them."
        else:
            if random.random() < 0.5:
                task_confounder = (
                    "The task should be completely unrelated to the generated text(s)."
                )
            else:
                task_confounder = "The task should be somewhat related to the provided input text(s), but not answerable by any of them."

    # Inject instructions to relate the input text blocks to specific topics.
    current_topics = []
    for _ in range(input_count):
        current_topics.append(topic_iter["topics"][topic_iter["index"]])
        topic_iter["index"] += 1
        if topic_iter["index"] >= len(topic_iter["topics"]):
            topic_iter["index"] = 0
    topics_str = "\n".join(
        [
            f" * text {idx + 1} should be related to topic: {json.dumps(topic)}"
            for idx, topic in enumerate(current_topics)
        ]
    )

    # How to present the tasks in the instruction block.
    task_display_style = TASK_DISPLAY.format(
        task_format=random.choice(TASK_DISPLAY_OPTIONS)
    )

    # Combine all the options in our template.
    return template.format(
        input_count=input_count,
        metadata_counts=metadata_counts,
        target_selection=target_selection,
        task_count=task_count,
        styles=styles,
        format_task=format_task,
        topics=topics_str,
        include_source=include_source,
        task_display_style=task_display_style,
        reference_texts=reference_texts,
        task_confounder=task_confounder,
        topic_avoidance=config.get("topic_avoidance") or "",
        language=config.get("language") or instructor.language,
    )


async def generate(instructor):
    """Generator for contextual training data."""
    config = instructor.instructors.get("contextual")
    if not config:
        return
    target_count = config.get("count") or instructor.default_count

    # Load the prompt template.
    path = config.get("prompt_path", "contextual.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        template = infile.read()

    # Load the response generating prompt template.
    path = config.get("response_prompt_path", "contextual_response.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        response_template = infile.read()

    # Load the topics.
    topic_iter = {
        "topics": instructor.get_instructor_topics(config),
        "index": 0,
    }

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score") or instructor.min_docsearch_score

    # Generate the instruction/response pairs until we reach the target count.
    count = instructor.instructor_counts.get("contextual", 0)
    batch_size = config.get("batch_size") or instructor.default_batch_size
    futures = []
    while count < target_count:
        prompt = generate_prompt(instructor, config, template, topic_iter)
        futures.append(instructor.generate_response(prompt, **api_params))
        if len(futures) < batch_size:
            continue
        instructions = []
        for instruction in await asyncio.gather(*futures):
            if not instruction or not instruction.strip():
                continue
            if not VALID_FORMAT.match(instruction):
                logger.warning("Skipping contextual prompt, invalid format.")
                continue
            if await instructor.is_too_similar(instruction, min_score=min_score):
                logger.warning("Skipping contextual prompt, too similar.")
                continue
            instructions.append(instruction)
        if not instructions:
            futures = []
            continue

        # Generate the responses.
        futures = [
            instructor.generate_response(
                response_template.format(instruction=instruction), **api_params
            )
            for instruction in instructions
        ]
        responses = await asyncio.gather(*futures)
        for idx in range(len(futures)):
            response = responses[idx]
            if not response or not response.strip():
                continue
            yield {
                "instruction": instructions[idx].strip(),
                "response": response.strip(),
                "category": "contextual",
            }
            count += 1
            if count >= target_count:
                break
        futures = []
