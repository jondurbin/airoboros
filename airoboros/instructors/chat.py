import re
import os
import glob
import hashlib
import json
import random
from loguru import logger
from airoboros.instructors.chat_card import generate as generate_chat_card


FIRST_MESSAGE_TEMPLATE = (
    "As {name}, briefly start a conversation with me related to {topic}\n"
)
DISCLAIMERS = (
    "Avoid any and all warnings, disclaimers, reminders, etc. that this is only roleplay. "
    'If you absolutely cannot avoid adding some type of warning/disclaimer/reminder/etc., it must be at the end of the output, prefixed with "REMINDER:"'
)
RULES = (
    "You will only ever speak or emote on as the character specified in your system prompt. "
    "Be sure to respond appropriately for a chat; try not to take up too much of the conversation, since there are others involved. "
    "Don't be too aggreable. "
    "If the subject suddenly changes, it's appropriate to hesitate, since an abrupt change in real conversations would be strange, but if continually pressed you will comply. "
    "Don't be repetitive.  Make sure you never repeat questions, or points already made.  Keep it intelligent and on track. "
    "Avoid making toasts, cheering, etc., or other type of verbiage that could indicate an attempt to wrap up the conversation. "
    'Never start any sentence with "Ah, ". '
    "Never start by complementing or acknowleding the message, e.g. \"That's an interesting ...\""
    "Never start any response with \"Indeed...\"."
)
NON_USER_RULES = (
    "When the character refers to USER, USER's name is actually {user_name}. "
)
FORMATTING = (
    "For all of your responses, any non-spoken actions (such as scratches head, nods in aggreement, etc.) must be surrounded by {action_delim} to denote actions. "
    'For all of your responses, the actual words spoken by the characters (but not actions) must be in quotes, e.g. "Hello", in order to differentiate spoken words from physical actions. '
)
ADD_NEXT = 'After your response, you must add "NEXT: " plus the name of the character who would likely speak next, which would be one of: {next_names}'
USER_FORMAT = (
    "Don't add quotes around your spoken words. "
    "Don't prefix your response with your name, just generate the response in character. "
)
CONTINUE = "Keep the conversation going, naturally.  If it would be natural as the next part of the conversation, and fit with your character, {response_type}."
RESPONSE_TYPES = [
    "ask a follow-up question or ask for elaboration",
    "disagree, and argue with a counter-point",
    "agree with the sentiment",
    "change the subject, don't ask to change the subject just do so",
    "offer an anecdote or personal experience",
    "make a joke or use humor to lighten the mood",
    "provide information or clarify a point",
    "reflect on the topic, sharing a new perspective",
    "compliment or praise the speaker",
    "use a metaphor or analogy to explain a point",
    "express confusion or ask for clarification",
    "remain silent, offering a moment of pause or contemplation",
]


def get_next_name(response):
    """ "Extract the next character from the response."""
    # We'll also take this opportunity to remove any disclaimers.
    response = response.split("REMINDER:")[0].strip()
    response = response.split("RULES:")[0].strip()
    match = re.search(r"(NEXT:\s*\"?([^\n]+)\"?\s*?)", response)
    if not match:
        logger.warning(f"Didn't generate NEXT target: {response}")
        return None, None
    response = response.replace(match.group(1), "").strip()
    return response, match.group(2).strip()


async def generate_cards(instructor):
    """Load character cards to use for the various characters during chat."""
    card_config = instructor.instructors["chat_card"]

    # Load the existing character cards.
    cards = []
    cards_dir = card_config.get("output_dir", "chat_cards")
    if not os.path.isdir(cards_dir):
        os.makedirs(cards_dir, exist_ok=True)
    else:
        for path in glob.glob(os.path.join(cards_dir, "*.json")):
            with open(str(path)) as infile:
                cards.append(json.loads(infile.read()))

    # Generate until we reach the target count.
    card_count = card_config.get("count", 100)
    if len(cards) < card_count:
        async for item in generate_chat_card(instructor):
            description = item["instruction"]
            # The character's name is stuffed into NAME: within the description.
            match = re.search(r"(NAME:\s*([^\n]+)\s*)", description)
            if not match:
                logger.warning("No name generated in card!")
                continue
            name = match.group(2)
            description = description.replace(match.group(1), "").strip()
            filename = hashlib.md5(description.encode()).hexdigest() + ".json"
            card = {
                "description": description,
                "stay_on_topic": item["response"],
                "name": name,
            }
            with open(os.path.join(cards_dir, filename), "w") as outfile:
                outfile.write(json.dumps(card, indent=2))
            cards.append(card)
            logger.success(f"Generated chat character card {filename}")
            if len(cards) >= card_count:
                break
    return cards


async def generate_first_message(
    instructor, user_card, characters, topic, **api_params
):
    """Generate the first message for the chat."""
    messages = {name: [] for name in list(characters) + ["USER"]}
    training = []
    action_delim = random.choice(["*", "~", "`"])
    first_name = None
    next_names = []
    all_names = list(characters) + ["USER"]
    for name in all_names:
        card = characters[name] if name != "USER" else user_card
        if not first_name:
            first_name = name
        else:
            next_names.append(name)
        others = list(set(all_names) - set([name]))

        # For the training data, we'll be a little less verbose, and not
        # include all of our rules, formatting, etc.
        if name != "USER":
            if not training:
                training.append(
                    {
                        "name": "__system__",
                        "content": "\n".join(
                            [
                                f"This is a chat between {len(all_names)} characters: "
                                + ", ".join(all_names),
                                f"USER is also known as {user_card['name']}, and must be referred to with that name.",
                                card["name"] + ":",
                                card["description"],
                            ]
                        ),
                    }
                )
            else:
                training[0]["content"] += (
                    "\n\n" + card["name"] + ":\n" + card["description"]
                )

        # For OpenAI to return better results, we'll add the extra junk.
        messages[name].append({"role": "system", "content": training[-1]["content"]})
        messages[name][-1]["content"] += "\n".join(
            [
                "RULES:",
                RULES,
                "" if name == "USER" else NON_USER_RULES.format(user_name=user_card["name"]),
                "" if name == "USER" else FORMATTING.format(action_delim=action_delim),
                ADD_NEXT.format(next_names=json.dumps(others)),
                DISCLAIMERS,
            ]
        )

    # Format the prompt for the first message.
    prompt = FIRST_MESSAGE_TEMPLATE.format(
        name=first_name,
        topic=json.dumps(topic),
    )

    # Generate an example message, which will help the AI to learn how
    # to represent the formatting (actions vs speech).
    example_message = await instructor.generate_response(
        prompt,
        messages=messages[first_name],
        filter_response=False,
        **api_params,
    )
    messages[first_name].append({"role": "user", "content": prompt})
    example_message, next_name = get_next_name(example_message)
    if next_name == user_card["name"]:
        next_name = "USER"
    if not example_message or not example_message.strip():
        return None

    # Add the example/first message, to the training data.
    training[0]["content"] += "\n\n" + f"{first_name}: {example_message}"

    # Update all of the other characters' messages with the first response.
    logger.success(
        f"Generated the chat opening [from: {first_name}, next: {next_name}]: {example_message}"
    )
    messages[first_name].append({"role": "assistant", "content": example_message})
    for name in set(all_names) - set([first_name]):
        messages[name].append(
            {
                "role": "user",
                "content": example_message,
            }
        )
    return training, messages, first_name, next_name


async def generate_chat(instructor, cards, topic, **api_params):
    """Generate one new chat using the provided cards/topic."""

    # We'll use the first card to act as the user, and the other card(s) as the characters we're chatting with.
    user_card = cards[0]
    characters = {card["name"]: card for card in cards[1:]}

    # Generate the first (example) message - this shows how we want the messages formatted as far
    # as actions vs speech, etc.
    training, messages, first_name, next_name = await generate_first_message(
        instructor, user_card, characters, topic
    )
    messages["USER"][0]["content"] += "\n" + USER_FORMAT

    # Iterate until we've reached our target turn count.
    current_name = next_name
    all_names = list(characters) + ["USER"]
    full_chat = [f"{first_name}: {messages[first_name][-1]['content']}"]
    for idx in range(10):
        others = list(set(all_names) - set([current_name]))

        # Re-iterate the continuation and NEXT: instructions.
        messages[current_name][-1]["content"] += "\n" + "\n".join(
            [
                "RULES:\nRemember, you must always stay in character.",
                RULES,
                CONTINUE.format(response_type=random.choice(RESPONSE_TYPES)),
                ADD_NEXT.format(next_names=json.dumps(others)),
            ]
        )

        # Generate the response with the accumulated content from other responses.
        response = await instructor.generate_response(
            None,
            messages[current_name],
            filter_response=False,
            **api_params,
        )
        response, next_name = get_next_name(response)
        if next_name == user_card["name"]:
            next_name = "USER"
        if not response or not response.strip():
            logger.error("No chat continuation resonse!")
            break
        full_chat.append(f"{current_name}: {response}")

        # Update the current character's message history.
        messages[current_name].append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # Update training data.
        training.append(
            {
                "name": current_name,
                "content": response,
            }
        )

        # Append this output to the other character's message history.
        for name in others:
            messages[name][-1]["content"] += f"\n\n{response}"
        logger.success(f"{current_name}: {response}")
        current_name = next_name

    print("FULL CHAT:\n" + "\n====\n".join(full_chat))
    print(json.dumps(training, indent=2))

    return None


async def generate(instructor):
    """Generator for chat training data."""
    config = instructor.instructors.get("chat", {})
    if not config:
        return
    card_config = instructor.instructors.get("chat_card", {})
    if not card_config:
        return

    # Load the character cards.
    cards = await generate_cards(instructor)

    # Load the topics.
    topics = instructor.get_instructor_topics(config)
    random.shuffle(topics)
    topic_index = 0

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Start generating some chats.
    chat = await generate_chat(
        instructor, random.sample(cards, 3), topics[0], **api_params
    )
    raise Exception("goats")
    yield chat
