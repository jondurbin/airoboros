from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor):
    """Generator for orca training data."""
    async for item in generate_inline(instructor, "orca"):
        yield item
