from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor):
    """Generator for rewoo style planning."""
    async for item in generate_inline(instructor, "plan"):
        yield item
