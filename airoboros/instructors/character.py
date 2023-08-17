import glob
import os
from airoboros.instructors.inline_qa import generate as generate_inline


async def generate(instructor, skip):
    """Generator for character card training data."""
    config = instructor.instructors.get("character", {})
    if not config:
        return

    # Load seed data.
    seed_path = config.get("seed_path", "character_seeds")
    if not os.path.isdir(seed_path):
        seed_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prompts", seed_path
        )
        if not os.path.isdir(seed_path):
            raise Exception("No seeds!")
    seeds = []
    for path in glob.glob(os.path.join(seed_path, "*.txt")):
        with open(str(path)) as infile:
            seeds.append(infile.read())
    seed_iter = 0

    def get_example(_):
        nonlocal seed_iter
        result = seeds[seed_iter]
        seed_iter += 1
        if seed_iter == len(seeds):
            seed_iter = 0
        return result

    template_kwargs = {"example": get_example, "skip": skip}
    async for item in generate_inline(
        instructor,
        "character",
        start_key="DESCRIPTION",
        end_key="GUIDE",
        filter_response=False,
        template_kwargs=template_kwargs,
    ):
        yield item
