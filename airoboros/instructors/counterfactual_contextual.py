import asyncio
import os
import random
import re
from loguru import logger


ADD_SOURCES = [
    "Provide a reference.",
    "Cite your source.",
    "What is your source?",
    "Source?",
    "Citation?",
    "[references]",
    "[citation]",
    "Add your source.",
    "Where is this information from?",
    "What was the link you find this in?",
]


async def generate(instructor):
    """Generator for contextual training data."""
    config = instructor.instructors.get("counterfactual_contextual")
    if not config:
        return
    target_count = config.get("count")
    if target_count is None:
        target_count = instructor.default_count
    target_count = int(target_count)
    if not target_count:
        return

    # Load the prompt template.
    path = config.get("prompt_path", "counterfactual_contextual.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        template = infile.read()

    # Load the response generating prompt template.
    path = config.get("response_prompt_path", "counterfactual_contextual_response.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        response_template = infile.read()

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score")
    if min_score is None:
        min_score = instructor.min_docsearch_score
    min_score = float(min_score)

    # Generate the instruction/response pairs until we reach the target count.
    count = instructor.instructor_counts.get("counterfactual_contextual", 0)
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    language = config.get("language") or instructor.language
    while count < target_count:
        response = await instructor.generate_response(
            template.format(batch_size=batch_size, language=language), **api_params
        )
        if not response or not response.strip():
            continue
        match = re.search("FACTS(.*)COUNTER(.*)QUESTIONS(.*)", response, re.DOTALL)
        if not match:
            logger.warning("Bad response format for counterfactual contextual prompt.")
            continue

        # Parse the response.
        facts = {}
        counter = {}
        questions = {}
        sources = {}

        # Find all of the factoids first.
        instructions = []
        futures = []
        for question_id, fact in re.findall(
            r"(?:^|\n)(\d+)\. (.*?)(?:$|(?=\n\d+\. ))", match.group(1), re.DOTALL
        ):
            facts[question_id] = fact

            # Extract the source URL, if provided.
            m = re.search(r"(\[(?:\w+:\s*)?(https.*?)\])", fact)
            if m:
                facts[question_id] = fact.replace(m.group(1), "")
                sources[question_id] = m.group(2)

            # Extract the counter/fake facts.
            for question_id, counter_fact in re.findall(
                r"(?:^|\n)(\d+)\. (.*?)(?:$|(?=\n\d+\. ))", match.group(2), re.DOTALL
            ):
                counter[question_id] = counter_fact

            # Extract the questions/instructions.
            for question_id, question in re.findall(
                r"(?:^|\n)(\d+)\. (.*?)(?:$|(?=\n\d+\. ))", match.group(3), re.DOTALL
            ):
                questions[question_id] = question

            # Ensure we have facts, counter facts, and questions for each question ID.
            for question_id in facts:
                if (
                    question_id not in questions
                    or question_id not in sources
                    or question_id not in counter
                ):
                    logger.warning(
                        f"Counterfactual contextual prompt response missing aspects of question {question_id}"
                    )
                    continue

                # Generate the prompt for both the actual factual statements and fake facts.
                for idx, target in enumerate([facts, counter]):
                    instruction = "\n".join(
                        [
                            "BEGININPUT",
                            "BEGINCONTEXT",
                            f"url: {sources[question_id]}",
                            "ENDCONTEXT",
                            target[question_id],
                            "ENDINPUT",
                            "BEGININSTRUCTION",
                            questions[question_id],
                            random.choice(ADD_SOURCES)
                            if random.random() <= 0.5
                            else "",
                            "ENDINSTRUCTION",
                        ]
                    )
                    if idx == 0 and await instructor.is_too_similar(
                        instruction, min_score=min_score
                    ):
                        logger.warning(
                            "Skipping counterfactual contextual instruction, too similar."
                        )
                        continue
                    instructions.append(instruction)
                    futures.append(
                        instructor.generate_response(
                            response_template.format(
                                instruction=instruction, language=language
                            ),
                            **api_params,
                        )
                    )

        # Generate responses.
        if not futures:
            continue
        responses = await asyncio.gather(*futures)
        for idx in range(len(futures)):
            response = responses[idx]
            if not response or not response.strip():
                continue
            yield {
                "instruction": instructions[idx].strip(),
                "response": response.strip(),
                "category": "counterfactual_contextual",
            }
            count += 1
            if count >= target_count:
                break
        futures = []
