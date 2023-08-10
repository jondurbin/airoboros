from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor, **kwargs):
    """Generator for riddle training data."""
    async for item in generate_inline(instructor, "riddle", **kwargs):
        yield item
