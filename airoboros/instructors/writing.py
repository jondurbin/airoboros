import math
import random
from airoboros.instructors.simple_task import generate as generate_simple_task


def generate_style_extra(instructor):
    """Inject a list of style directives."""
    batch_size = instructor.instructors["writing"].get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    with_styles = math.floor(batch_size / 4)
    if not with_styles and batch_size > 1:
        with_styles = 1
    if with_styles > len(instructor.instructors["writing"]["styles"]):
        with_styles = instructor.instructors["writing"]["styles"]
    batch_styles = random.sample(
        instructor.instructors["writing"]["styles"], with_styles
    )
    return "Additional requirements:\n" + "\n".join(
        [
            f"- instruction {idx + 1} should specify that the style be {batch_styles[idx]}"
            for idx in range(with_styles)
        ]
    )


async def generate(instructor, **kwargs):
    """Generator for writing training data."""
    conf = instructor.instructors.get("writing", {})
    if not conf:
        return
    styles = conf.get("styles", [])
    template_kwargs = {}
    if styles:
        template_kwargs["style_extra"] = generate_style_extra
    else:
        template_kwargs["style_extra"] = lambda _: ""
    async for item in generate_simple_task(
        instructor, "writing", template_kwargs=template_kwargs, **kwargs
    ):
        yield item
