from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor):
    """Generator for chain-of-thought training data."""
    async for item in generate_inline(instructor, "cot"):
        yield item
