import random
from loguru import logger
from airoboros.instructors.inline_qa import generate as generate_inline


ALL_OPTIONS = "* QUESTION {index} must have options: A, B, C, D, and E"
FOUR_OPTIONS = "* QUESTION {index} must have options: A, B, C, and D"
COMBINATORIAL = "where E must be a value similar to 'None of the above', 'All of the above', 'A and C', or other combinatorial value"
CONTEXTUAL = (
    "* for QUESTION {index}, please generate a complex paragraph consisting of at least 4 sentences related to the specified topic, "
    "then generate a question related to the paragraph, such as classification, contextual question answering, etc., "
    "but don't make it obvious - the question's answer shouldn't be directly included in the text verbatim"
)


async def generate(instructor, **kwargs):
    """Generator for multiple choice training data."""
    config = instructor.instructors.get("multiple_choice")
    if not config:
        return
    flesch = config.get("flesch") or "40"
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)

    current_index = 0
    options = ["A", "B", "C", "D", "E"]

    # Helper to make sure the answers aren't always B/C (gpt-4 tends to not distribute well).
    def gen_answer_index(_):
        nonlocal current_index
        nonlocal options
        answer_options = []
        for idx in range(batch_size):
            answer_options.append(
                f"* the correct option for QUESTION {idx + 1} must be {options[current_index]}"
            )
            current_index += 1
            if current_index == len(options):
                current_index = 0
        return "\n".join(answer_options)

    current_option_index = 0

    # Helper to sometimes include option E.
    def gen_options(_):
        nonlocal current_option_index
        nonlocal options
        current_options = []
        for idx in range(batch_size):
            if current_option_index == len(options) - 1:
                current_options.append(ALL_OPTIONS.format(index=str(idx + 1)))
            else:
                if random.random() <= 0.3:
                    current_options.append(ALL_OPTIONS.format(index=str(idx + 1)))
                    current_options[-1] += f", {COMBINATORIAL}"
                else:
                    current_options.append(FOUR_OPTIONS.format(index=str(idx + 1)))
            current_option_index += 1
            if current_option_index == len(options):
                current_option_index = 0
        return "\n".join(current_options)

    # Helper to sometimes include context snippet.
    def gen_context(_):
        nonlocal config
        contextual = []
        for idx in range(batch_size):
            if random.random() <= float(config.get("contextual_ratio") or "0.2"):
                contextual.append(CONTEXTUAL.format(index=str(idx + 1)))
        return "\n".join(contextual)

    async for item in generate_inline(
        instructor,
        "multiple_choice",
        template_kwargs={
            "answers": gen_answer_index,
            "options": gen_options,
            "context": gen_context,
            "flesch": lambda _: flesch,
        },
    ):
        # Let's double check the answer letter/text match.
        if item["response"].strip() not in item["instruction"]:
            logger.warning(f"Response mismatch: {item['response']}")
            continue
        yield item
