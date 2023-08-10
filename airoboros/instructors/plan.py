from airoboros.instructors.simple_task import generate as generate_simple


async def generate(instructor, **kwargs):
    """Generator for rewoo style planning."""
    async for item in generate_simple(
        instructor, "plan", filter_response=False, **kwargs
    ):
        yield item
