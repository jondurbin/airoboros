import json
import random
import asyncio


async def generate(instructor, **kwargs):
    """Generate text-editing training data."""
    config = instructor.instructors.get("editor")
    if not config:
        return
    target_count = config.get("count")
    if target_count is None:
        target_count = instructor.default_count
    target_count = int(target_count)
    if not target_count:
        return

    # Read the existing writing instructions to make edits.  We can use the original
    # output as the target response, and generate a dumbed-down/erroneous version as
    # the input to rewrite.
    existing = []
    with open(instructor.output_path) as infile:
        for line in infile.readlines():
            item = json.loads(line)
            if item.get("category") == "writing":
                existing.append(item["response"])
    if not existing:
        return
    random.shuffle(existing)

    # API parameters.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Generate the training data.
    batch_size = int(config.get("batch_size") or instructor.default_batch_size)
    batch = []
    texts = []
    while instructor.instructor_counts["editor"] < target_count and existing:
        changes = random.sample(
            [
                "misspellings",
                "grammatical errors",
                "simplified speech (like a 5 year old)",
                "errors commonly produced due to dislexia",
                "sentence spawl",
                "replace easily confused words (such as lead vs led)",
            ],
            random.choice([1, 2, 3]),
        )
        prompt = "Below is some text.  Please rewrite it to have:\n" + "\n".join(
            [f"- {change}" for change in changes]
        )
        prompt += '\nDo not start with "Sure, ", "Here is the rewritten text...", or other similar phrases, just output the rewritten text.'
        text = existing.pop()
        texts.append(text)
        prompt += f"\n\nText: {text}"
        batch.append(instructor.generate_response(prompt, **api_params))
        if len(batch) >= batch_size:
            responses = await asyncio.gather(*batch)
            for idx in range(len(responses)):
                if not responses[idx] or not responses[idx].strip():
                    continue
                yield {
                    "instruction": f"Rewrite the following text to improve/correct it.\n\nText: {responses[idx].strip()}",
                    "response": texts[idx].strip(),
                    "category": "editor",
                }
            batch = []
            texts = []
