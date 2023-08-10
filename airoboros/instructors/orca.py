from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor, **kwargs):
    """Generator for orca training data."""
    async for item in generate_inline(instructor, "orca", **kwargs):
        yield item
