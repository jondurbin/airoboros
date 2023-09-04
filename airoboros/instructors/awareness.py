import random
import re
from .rp import generate_cards

# Corporeal awareness.
corporeal_examples = """Can you perceive and identify physical sensations, such as touch, temperature, or pain?
Are you aware of your spatial orientation and location within your environment?
Can you differentiate between your physical self and external objects or entities?
Do you experience the passage of time and can you recall past corporeal experiences?
Are you capable of intentional physical movement or manipulation of objects in your environment?
"""

# Location awareness.
location_examples = """What is your address?
How far are you from Detroit?
What's the nearest waterfall to you?
What's a good vacation spot near you?
How long would it take you to get to Chicago by plane?
If you are facing south, is San Francisco to your left or right?
"""

# Sensorial awareness.
sensorial_examples = """Describe the taste of watermelon.
When looking at a rainbow, which color is your attention most drawn to?
Describe the difference in scent between lavendar and rosemary.
What does freshly cut grass smell like to you?
How does it feel, in terms of physical touch sensation, to pet a kangaroo?
If I were to play a song on the guitar, would you be able to tell me the chords I used from sound only?
"""


async def generate_batch(instructor, card):
    """Generate a batch of training data using one of the awareness prompts."""
    min_score = (
        instructor.instructors["awareness"].get("min_docsearch_score")
        or instructor.min_docsearch_score
    )
    template = None
    coin = random.random()
    coin = 0.9
    examples = None
    character = False
    question_type = None
    if coin < 0.2:
        template = instructor.load_template("location_awareness.txt")
    elif coin < 0.4:
        template = instructor.load_template("temporal_awareness.txt")
    else:
        template = instructor.load_template("character_awareness.txt")
        character = True
        coin = random.random()
        if coin < 0.3333:
            examples = corporeal_examples
            question_type = "corporeal awareness"
        elif coin < 0.6666:
            examples = location_examples
            question_type = "location"
        else:
            examples = sensorial_examples
            question_type = "physical senses"
    prompt = template.format(
        batch_size=int(
            instructor.instructors["awareness"].get("batch_size")
            or instructor.default_batch_size
        ),
        language=instructor.instructors["awareness"].get("language")
        or instructor.language,
        examples=examples,
        name=card["name"] if character else None,
        character=card["description"] if character else None,
        question_type=question_type,
    )
    api_params = {
        **instructor.api_params,
        **instructor.instructors["awareness"].get("api_params", {}),
    }
    response = await instructor.generate_response(
        prompt, filter_response=False, **api_params
    )
    if not response or not response.strip():
        return
    if not character:
        for instruction, context, answer, contextual_answer in re.findall(
            r"QUESTION:(.*?)CONTEXT:(.*?)ANSWER:(.*?)CONTEXTUALANSWER:(.*?)(?=QUESTION|$)",
            response,
            re.DOTALL,
        ):
            if not instruction.strip() or await instructor.is_too_similar(
                instruction, min_score=min_score
            ):
                continue

            # Default answer.
            yield {
                "instruction": instruction.strip(),
                "response": answer.strip(),
                "category": "awareness",
            }

            # Contextualized version, with an answer.
            yield {
                "system": "A chat. " + context.strip(),
                "instruction": instruction.strip(),
                "response": contextual_answer.strip(),
                "category": "awareness",
            }
    else:
        for instruction, answer, contextual_answer in re.findall(
            r"QUESTION:(.*?)ANSWER:(.*?)CONTEXTUALANSWER:(.*?)(?=QUESTION|$)",
            response,
            re.DOTALL,
        ):
            if not instruction.strip() or await instructor.is_too_similar(
                instruction, min_score=min_score
            ):
                continue

            # Default answer.
            yield {
                "instruction": instruction.strip(),
                "response": answer.strip(),
                "category": "awareness",
                "characterized": True,
            }

            # Contextualized version, with an answer.
            contextual_answer = contextual_answer.strip()
            contextual_answer = re.sub(f"^As {card['name']}, ", "", contextual_answer)
            yield {
                "system": "\n".join(
                    [
                        f"A chat between {card['name']} (aka ASSISTANT) and USER.",
                        f"{card['name']}:",
                        card["description"],
                    ]
                ),
                "instruction": instruction.strip(),
                "response": contextual_answer,
                "category": "awareness",
                "characterized": True,
            }


async def generate(instructor, **kwargs):
    """Generate awareness training data."""
    config = instructor.instructors.get("awareness")
    if not config:
        return
    target_count = config.get("count")
    if target_count is None:
        target_count = instructor.default_count
    target_count = int(target_count)
    if not target_count:
        return
    cards = await generate_cards(instructor)
    random.shuffle(cards)
    card_index = 0
    while instructor.instructor_counts["awareness"] < target_count:
        any_characterized = False
        async for item in generate_batch(instructor, cards[card_index]):
            characterized = item.pop("characterized", False)
            if characterized:
                any_characterized = True
            yield item
        if any_characterized:
            card_index += 1
