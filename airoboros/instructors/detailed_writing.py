import asyncio
import glob
import json
import os
import random
from loguru import logger

INNER_PART = (
    "Now, generate the {index} part, which must include a minimum of {third} words. "
    "Remember, this is only the {index} part, and the remaining sections will be asked for later. "
    "Don't try to add some sort of conclusion or wrap-up sentence or paragraph, because this is only the {index} part. "
    'The output for this section should end in such a way as to be easily and fluidly continued by a subsequent prompt, i.e. don\'t end with "And so, ... " '
    "Don't include any indication that this is only one part, e.g. 'to be continued..', etc., just output the {index} part."
)
FINAL_PART = "Generate the final part, which must include a mimimum of {third} words."
COMBINE = (
    "Below is an instruction, and the response to the instruction. "
    "Please read the instruction very carefully, then read the response. "
    "I would like you rewrite the response have a better flow to it. "
    "Use a broader, more colorful and vibrant vocabulary, and add in interesting details. "
    "Make sure all of the requirements are met, and remove any hallucinated factors not outlined in the requirements. "
    "The most important requirement is this: do not shorten the overall length.  The output must be at least as long as the original response.\n\n"
    "Instruction: {instruction}\n\nResponse: {response}"
)


async def gen_with_retry(instructor, prompt, messages=[], attempt=0, **api_params):
    result = await instructor.generate_response(
        prompt, messages=messages, filter_response=False, **api_params
    )
    if result and result.strip():
        return result
    if attempt > 5:
        return None
    return await gen_with_retry(
        instructor, prompt, messages=messages, attempt=attempt + 1, **api_params
    )


async def generate(instructor, **kwargs):
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

    # Load the prompt templates.
    template = instructor.load_template(
        config.get("prompt_path", "detailed_writing.txt")
    )
    response_template = instructor.load_template(
        config.get("response_prompt_path", "detailed_writing_response.txt")
    )

    # Load the topics.
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
    if "detailed_writing" not in instructor.instructor_counts:
        instructor.instructor_counts["detailed_writing"] = 0
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    futures = []
    language = config.get("language") or instructor.language
    flesch = config.get("flesch") or instructor.default_flesch
    word_count = config.get("word_count", 4000)
    while instructor.instructor_counts["detailed_writing"] < target_count:
        # Generate a unique prompt to use.
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

        # Once we have batch_size instruction prompts, we can generate responses,
        # which are the actual instructions.
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

        # Generate the first part of the response.
        partitioned_instructions = [
            response_template.format(
                instruction=instruction,
                flesch=flesch,
                language=language,
                word_count=int(word_count * 1.3),  # words != tokens, add buffer
                third=int((word_count * 1.3) / 3),
            )
            for instruction in instructions
        ]
        futures = [
            instructor.generate_response(
                instruction, filter_response=False, **api_params
            )
            for instruction in partitioned_instructions
        ]
        responses = await asyncio.gather(*futures)

        # Generate the second part of the response.
        messages = []
        futures = []
        original_instructions = []
        for idx in range(len(responses)):
            if not responses[idx] or not responses[idx].strip():
                continue
            original_instructions.append(instructions[idx])
            messages.append(
                [
                    {
                        "role": "user",
                        "content": partitioned_instructions[idx],
                    },
                    {
                        "role": "assistant",
                        "content": responses[idx],
                    },
                ]
            )
            futures.append(
                gen_with_retry(
                    instructor,
                    INNER_PART.format(
                        index="second", third=int((word_count * 1.3) / 3)
                    ),
                    messages=messages[-1],
                    attempt=0,
                    **api_params,
                )
            )
        if not futures:
            continue

        # Generate the final section of the response.
        responses = await asyncio.gather(*futures)
        messages_final = []
        futures = []
        successful_instructions = []
        for idx in range(len(responses)):
            if not responses[idx] or not responses[idx].strip():
                continue
            successful_instructions.append(original_instructions[idx])
            messages_final.append(
                messages[idx]
                + [
                    {
                        "role": "user",
                        "content": INNER_PART.format(
                            index="second", third=int((word_count * 1.3) / 3)
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": responses[idx],
                    },
                ]
            )
            futures.append(
                gen_with_retry(
                    instructor,
                    FINAL_PART.format(third=int((word_count * 1.3) / 3)),
                    messages=messages_final[-1],
                    attempt=0,
                    **api_params,
                )
            )
        if not futures:
            continue
        responses = await asyncio.gather(*futures)

        # Now put everything together.
        fluid_instructions = []
        futures = []
        for idx in range(len(responses)):
            response = responses[idx]
            if not response or not response.strip():
                continue
            full_response = []
            for message in messages_final[idx]:
                if message["role"] == "assistant":
                    full_response.append(message["content"].strip())
            full_response.append(response)
            fluid_instructions.append(successful_instructions[idx].strip())
            futures.append(
                gen_with_retry(
                    instructor,
                    COMBINE.format(
                        instruction=successful_instructions[idx],
                        response="\n\n".join(full_response),
                    ),
                    messages=[],
                    attempt=0,
                    **api_params,
                )
            )
        if not futures:
            continue
        fluid_responses = await asyncio.gather(*futures)
        for idx in range(len(futures)):
            if not fluid_responses[idx] or not fluid_responses[idx].strip():
                continue
            apprx_word_count = len(fluid_responses[idx].split(" "))
            yield {
                "instruction": fluid_instructions[idx].strip()
                + f"\n\nYour response should be approximately {apprx_word_count} words.",
                "response": fluid_responses[idx].strip(),
                "category": "detailed_writing",
            }
            if instructor.instructor_counts["detailed_writing"] >= target_count:
                break
        futures = []
