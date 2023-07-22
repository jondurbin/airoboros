from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor):
    """Generator for model/scenario card training data."""
    async for item in generate_inline(instructor, "card", start_key="PROMPT"):
        yield item
