from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor):
    """Generator for joke training data."""
    async for item in generate_inline(instructor, "joke"):
        yield item
