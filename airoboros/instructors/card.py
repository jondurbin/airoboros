from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor, **kwargs):
    """Generator for model/scenario card training data."""
    async for item in generate_inline(instructor, "card", start_key="PROMPT", **kwargs):
        yield item
