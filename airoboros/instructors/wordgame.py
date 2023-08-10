from airoboros.instructors.simple_task import generate as generate_simple_task


async def generate(instructor, **kwargs):
    """Generator for wordgame training data."""
    async for item in generate_simple_task(instructor, "wordgame", **kwargs):
        yield item
