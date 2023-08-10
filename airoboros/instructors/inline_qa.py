import json
import os
import random
import re


async def generate(
    instructor,
    category,
    start_key="QUESTION",
    end_key="ANSWER",
    filter_response=True,
    template_kwargs={},
    **kwargs,
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

    # Topics included in template?
    topics = instructor.get_instructor_topics(config)
    topic_index = random.randint(0, len(topics) - 1)

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
    if category not in instructor.instructor_counts:
        instructor.instructor_counts[category] = 0
    language = config.get("language") or instructor.language
    flesch = config.get("flesch") or instructor.default_flesch
    while instructor.instructor_counts[category] < target_count:
        # Get a batch of instructions.
        prompt_args = {"language": language, "flesch": flesch}
        if "{batch_size}" in template:
            prompt_args["batch_size"] = batch_size
        if "{topic_avoidance}" in template:
            prompt_args["topic_avoidance"] = instructor.topic_avoidance

        # Inject the topics to use for this batch, if enabled.
        if "{topics}" in template:
            current_topics = []
            for _ in range(batch_size):
                current_topics.append(topics[topic_index])
                topic_index += 1
                if topic_index >= len(topics):
                    topic_index = 0
            topics_str = "\n".join(
                [
                    f" * {start_key} {idx + 1} must be related to topic: {json.dumps(topic)}"
                    for idx, topic in enumerate(current_topics)
                ]
            )
            prompt_args["topics"] = topics_str

        # Inject any additional template args.
        for key, method in template_kwargs.items():
            prompt_args[key] = method(instructor)

        prompt = template.format(**prompt_args)
        response = await instructor.generate_response(
            prompt,
            messages=kwargs.get("messages", []),
            filter_response=filter_response,
            **api_params,
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
            if instructor.instructor_counts[category] >= target_count:
                break
