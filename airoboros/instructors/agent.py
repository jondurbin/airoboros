from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor):
    """Generator for agent/router training data."""
    async for item in generate_inline(
        instructor, "agent", start_key="PROMPT", filter_response=False
    ):
        yield item
