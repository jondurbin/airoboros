from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor, **kwargs):
    """Generator for joke training data."""
    async for item in generate_inline(instructor, "joke", **kwargs):
        yield item
