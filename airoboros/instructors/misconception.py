from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor, **kwargs):
    """Generator for misconception training data."""
    async for item in generate_inline(instructor, "misconception", **kwargs):
        yield item
