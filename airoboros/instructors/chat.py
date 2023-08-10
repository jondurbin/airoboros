import re
import os
import glob
import hashlib
import json
import random
from difflib import get_close_matches
from loguru import logger
from airoboros.instructors.chat_card import generate as generate_chat_card


FIRST_MESSAGE_TEMPLATE = "As {name}, briefly start the conversation."
DISCLAIMERS = """
Avoid any and all warnings, disclaimers, reminders, etc. that this is only roleplay.
If you absolutely cannot avoid adding some type of warning/disclaimer/reminder/etc., it must be at the end of the output, prefixed with "REMINDER:"
"""
RULES = """
The output must read like a roleplay chat, not a novel.
Give interesting, detailed, loquacious responses with several sentences, unless you are asked a simple, direct question.
Use normal ascii punctuation/special characters instead of utf-8.
You are {name}.  You will only ever speak, emote, or narrate as {name}.  You will never add responses, actions, etc. on behalf of other characters.
Don't be too aggreable - act as the character, hold to their beliefs, mood, etc., don't just abruptly change your entire philosophy.
Tune down your normal sunny disposition - not every conversation is cheery and fun.
If the subject suddenly changes, it's appropriate to hesitate, since an abrupt change in real conversations would be strange, but if continually pressed you will comply.
Don't be repetitive.  Make sure you never repeat questions, or points already made.
You must not repeat phrases or actions that have already appeared in the conversation.
Avoid making toasts, cheering, etc., or other type of verbiage that could indicate an attempt to wrap up the conversation.
Never start your response with an acknowedgement of the input, such as "Indeed, the idea of [..] is intriguing."
Never start any sentence with "Ah, ".
Never start by complementing or acknowleding the message, e.g. "That\'s an interesting ..."
Never start any response with "Indeed...".
Never speak in the third person or make a reference to your character's name in the response.
It is clear that you are the one performing the action/speaking, so you must never reference your character's name.
{flesch}
"""
NON_USER_RULES = """
All characters should speak in roughly equal quantities, however be sure to include USER slightly more often.
When the character refers to USER, USER's name is actually {user_name}.
Your response should be a minimum of 200 words - give long, detailed, immersive, colorful, insightful, and intelligent responses, and be sure to re-read the system prompt and follow the guidance and chat setting provided therein.
"""
FORMATTING = """
Actions the character performs must be differentiated from spoken words and general narration.
Actions must therefore be surround by {action_delim}, e.g. {action_delim}she slowly lifts her gaze{action_delim}
Keep the actions fairly succint and lowercase, and combine any immediately adjacent actions.
Characters must avoid repeating actions.
Actions will not include any references to the actor or include "I [some action]".
For example, instead of {action_delim}I raise an eyebrow{action_delim}, the action would be {action_delim}raises an eyebrow{action_delim}.
Never start the action with "I ..."
Words spoken by the character must be differentiated from actions and general narration.
Words spoken must therefore be in quotes, i.e. "[words spoken]"
General narration, that isn't a specific action or spoken word, must not be quoted or surrounded by {action_delim}.
General narration formatting should be used for general descriptions of the scene/backstory/etc, but not actions or spoken words.
"""
ADD_NEXT = """
After your response, you must add "NEXT: " plus the name of the character who would likely speak next, from the following list: {next_names}
Be sure to actually include a response too, not just the NEXT token.
"""
CONTINUE = """
Keep the conversation going in a natural, flowing way; {conv_turn}."
"""
CONV_TURNS = [
    "become completely fascinated and entirely absorbed by topic being discussed",
    "dive deep into the subject, with extraordinary detail; be immersive",
    "ask a follow-up question or ask for elaboration",
    "disagree, and argue with a counter-point",
    "change the subject to {topic} - don't ask to change the subject, just do it",
    "discuss or question how the current conversation might relate to {topic}",
    "offer an anecdote or personal experience",
    "make a joke or use humor to lighten the mood",
    "provide information or clarify a point",
    "share a new, personal perspective on the subject at hand",
    "use a metaphor or analogy to explain a point",
    "express confusion or ask for clarification",
    "ask about one of the other character's history/story, interests, or other personal information",
]


def parse_response(response, current_name, user_name, names, action_delim):
    """ "Extract the next character from the response."""
    # We'll also take this opportunity to remove any disclaimers.
    response = response.split("REMINDER:")[0].strip()
    response = response.split("RULES:")[0].strip()
    match = re.search(r"(NEXT:\s*([^\n]+))", response)
    name = "USER"
    if not match:
        logger.warning(f"Didn't generate NEXT target: {response}")
        if current_name == "USER":
            name = random.choice(list(set(names) - set(["USER"])))
    else:
        response = response.replace(match.group(1), "").strip()
        name = match.group(2).strip().replace('"', "")
    if response.startswith(f"{current_name}:"):
        response = response[len(current_name) + 1 :].lstrip()
    response = response.replace("USER", user_name)

    # Clean up any hallucinated responses on behalf of other names.
    other_names = set(names) - set([current_name])
    if current_name not in ("USER", user_name):
        other_names.add(user_name)
        other_names.add("USER")
    other_names_re = (
        "\n(" + "|".join([str(re.escape(name)) for name in other_names]) + "):"
    )
    response = re.split(other_names_re, response)[0]

    # Cleanup stray action delimiters.
    response = re.sub(
        f'({re.escape(action_delim)})([\\.,\\s-]*){re.escape(action_delim)}\\s*"',
        r'\1 "',
        response,
    )

    # Handle any misspellings or 's, etc. in case the name doesn't match.
    if name == user_name:
        name = "USER"
    elif name not in names:
        matches = get_close_matches(name, names)
        if not matches:
            name = random.choice(names)
    return response, name


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

    # Prevent duplicate names.
    names = set([card["name"] for card in cards])

    # Generate until we reach the target count.
    card_count = card_config.get("count", 100)
    skip = (
        lambda _: ""
        if not names
        else "Do not use any of the following for your new character: "
        + ", ".join(list(names))
    )
    if len(cards) < card_count:
        async for item in generate_chat_card(instructor, skip=skip):
            description = item["instruction"]
            # The character's name is stuffed into NAME: within the description.
            match = re.search(r"(NAME:\s*([^\n]+)\s*)", description)
            if not match:
                logger.warning("No name generated in card!")
                continue
            name = match.group(2)
            if name in names:
                logger.warning(f"Skipping duplicate name: {name}")
                continue
            names.add(name)
            description = description.replace(match.group(1), "").strip()
            filename = hashlib.md5(description.encode()).hexdigest() + ".json"
            card = {
                "description": description,
                "stay_in_character": item["response"],
                "name": name,
            }
            with open(os.path.join(cards_dir, filename), "w") as outfile:
                outfile.write(json.dumps(card, indent=2))
            cards.append(card)
            logger.success(f"Generated chat character card {filename}")
            if len(cards) >= card_count:
                break
    instructor.instructor_counts["chat_card"] = len(cards)
    return cards


async def generate_setting(instructor, user_card, characters, topic, **api_params):
    """Generate a setting to use for the chat."""
    path = (
        instructor.instructors["chat"].get("setting_prompt_path") or "chat_setting.txt"
    )
    prompt_template = instructor.load_template(path)
    return await instructor.generate_response(
        prompt_template.format(
            characters="\n\n".join(
                [
                    f"{name}: {card['description']}"
                    for name, card in {
                        **characters,
                        **{user_card["name"]: user_card},
                    }.items()
                ]
            ),
            topic=topic,
        ),
        **api_params,
    )


async def generate_first_message(
    instructor, user_card, characters, topic, **api_params
):
    """Generate the first message for the chat."""
    messages = {name: [] for name in list(characters) + ["USER"]}
    flesch = (
        instructor.instructors.get("chat", {}).get("flesch")
        or instructor.default_flesch
    )
    action_delim = random.choice(["*", "~", "`"])
    first_name = None
    all_names = list(characters) + ["USER"]
    setting = await generate_setting(
        instructor, user_card, characters, topic, **api_params
    )
    character_block = "\n\n".join(
        [
            f'{name}: {card["description"].strip()}'
            for name, card in {**characters, **{user_card["name"]: user_card}}.items()
        ]
    )
    training = [
        {
            "role": "system",
            "content": "\n\n".join(
                [
                    f"This is a chat between {len(all_names)} characters: "
                    + ", ".join(all_names),
                    character_block,
                    f"USER is also known as {user_card['name']}, and must be referred to with that name.",
                    f"Setting for the chat:\n{setting}\nEnd of setting.",
                ]
            ),
        }
    ]
    logger.success(f"Generated the chat card:\n{training[0]['content']}")
    formatting = FORMATTING.format(action_delim=action_delim)
    for name in all_names:
        if not first_name:
            first_name = name
        others = list(set(all_names) - set([name]))

        # Create the OpenAI messages to use for generating responses, which require some extra rules.
        hint = (
            characters[name]["stay_in_character"]
            if name != "USER"
            else user_card["stay_in_character"]
        )
        messages[name].append(
            {
                "role": "system",
                "content": training[0]["content"]
                + "\n"
                + "\n".join(
                    [
                        "RULES:",
                        RULES.format(
                            flesch=flesch,
                            name=user_card["name"] + " AKA USER"
                            if name == "USER"
                            else name,
                        ),
                        formatting,
                        (
                            ""
                            if name == "USER"
                            else NON_USER_RULES.format(user_name=user_card["name"])
                        ),
                        ADD_NEXT.format(next_names=json.dumps(others)),
                        DISCLAIMERS,
                        hint,
                        "Remember to always follow the specified formatting rules, regarding differentiation between spoken words, actions, and general narration.",
                    ]
                ),
            }
        )

    # Format the prompt for the first message.
    prompt = FIRST_MESSAGE_TEMPLATE.format(name=first_name)

    # Generate an example message, which will help the AI to learn how
    # to represent the formatting (actions vs speech).
    example_message = await instructor.generate_response(
        prompt,
        messages=messages[first_name],
        filter_response=False,
        **api_params,
    )
    messages[first_name].append({"role": "user", "content": prompt})
    example_message, next_name = parse_response(
        example_message, first_name, user_card["name"], all_names, action_delim
    )
    if not example_message or not example_message.strip():
        raise RuntimeError("Failed to generate example message")

    # Add the example/first message, to the training data.
    training[0]["content"] += (
        "\n\nStart of the conversation:\n" + f"{first_name}: {example_message}"
    )

    # Update all of the other characters' messages with the first response.
    logger.success(
        f"Generated the chat opening [from: {first_name}, next: {next_name}]: {example_message}"
    )
    messages[first_name].append({"role": "assistant", "content": example_message})
    for name in set(all_names) - set([first_name]):
        messages[name].append(
            {
                "role": "user",
                "content": f"{first_name}: {example_message}",
            }
        )
    return training, messages, first_name, next_name, action_delim


async def generate_chat(instructor, cards, topic, **api_params):
    """Generate one new chat using the provided cards/topic."""

    # We'll use the first card to act as the user, and the other card(s) as the characters we're chatting with.
    user_card = cards[0]
    characters = {card["name"]: card for card in cards[1:]}

    # Generate the first (example) message - this shows how we want the messages formatted as far
    # as actions vs speech, etc.
    try:
        (
            training,
            messages,
            first_name,
            next_name,
            action_delim,
        ) = await generate_first_message(instructor, user_card, characters, topic)
    except RuntimeError:
        return None

    # Iterate until we've reached our target turn count.
    current_name = next_name
    all_names = list(characters) + ["USER"]
    flesch = (
        instructor.instructors.get("chat", {}).get("flesch")
        or instructor.default_flesch
    )
    target_turns = instructor.instructors["chat"].get("turn_count") or 50
    topics = instructor.get_instructor_topics(instructor.instructors["chat"])
    user_name = user_card["name"]
    while True:
        others = list(set(all_names) - set([current_name]))

        # Continue the conversation, occasionally injecting random turns and topic changes.
        next_turn = random.choice(CONV_TURNS)
        if "{topic}" in next_turn:
            change_topic = random.choice(topics)
            while topic == change_topic:
                change_topic = random.choice(topics)
            next_turn = next_turn.format(topic=json.dumps(change_topic))

        messages[current_name][-1]["content"] += "\n" + "\n".join(
            [
                "RULES:\nRemember, you must always stay in character.",
                RULES.format(
                    flesch=flesch,
                    name=current_name
                    if current_name != "USER"
                    else f"{current_name} AKA {user_name}",
                ),
                CONTINUE.format(conv_turn=random.choice(CONV_TURNS)),
                ADD_NEXT.format(next_names=json.dumps(others)),
            ]
        )

        # Generate the response with the accumulated content from other responses.
        response = await instructor.generate_response(
            None,
            messages=messages[current_name],
            filter_response=False,
            **api_params,
        )
        if not response or not response.strip():
            logger.warning("No response, rerolling!")
            continue

        # We'll remove the re-iterated rules from the prompt to reduce token usage.
        messages[current_name][-1]["content"] = (
            messages[current_name][-1]["content"].split("RULES:")[0].strip()
        )

        response, next_name = parse_response(
            response, current_name, user_card["name"], all_names, action_delim
        )
        if not response or not response.strip():
            logger.warning("No chat continuation resonse, rerolling!")
            continue

        # Update the current character's message history.
        messages[current_name].append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # Update training data.
        prefix = "" if len(characters) == 1 else f"{current_name}: "
        if current_name != "USER" and training[-1]["role"] in ("system", "assistant"):
            training[-1]["content"] += f"\n\n{prefix}{response}"
        else:
            training.append(
                {
                    "role": "assistant" if current_name != "USER" else "user",
                    "content": f"{prefix}{response}"
                    if current_name != "USER"
                    else response,
                }
            )

        # Append this output to the other character's message history.
        for name in others:
            if messages[name][-1]["role"] == "assistant":
                messages[name].append({"role": "user", "content": ""})
            messages[name][-1]["content"] += f"\n\n{current_name}: {response}"
        logger.success(f"{current_name}: {response}")
        current_name = next_name
        if len(training) >= target_turns or (
            current_name == "USER" and len(training) >= target_turns - 1
        ):
            logger.success(f"Reached {len(training)}, finished.")
            break
    return training


async def generate(instructor, **kwargs):
    """Generator for chat training data."""
    config = instructor.instructors.get("chat", {})
    if not config:
        return
    card_config = instructor.instructors.get("chat_card", {})
    if not card_config:
        return
    target_count = instructor.instructors["chat"].get("count")
    if target_count is None:
        target_count = instructor.default_count
    if not target_count:
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
    while instructor.instructor_counts["chat"] < target_count:
        # Select a random number of characters.
        count_options = [2, 3, 4, 5]
        count_weights = [0.7, 0.15, 0.1, 0.05]
        card_count = random.choices(count_options, count_weights)[0]
        chat = await generate_chat(
            instructor,
            random.sample(cards, card_count),
            topics[topic_index],
            **api_params,
        )
        if not chat:
            continue

        # We'll convert each round into a row of training data, i.e.:
        # instruction 0 = system + user, response 0 = assistant response 0
        # instruction 1 = system + user + assistant + user, response 1 = assistant response 1
        # This way all of our existing training scripts should work without any changes.
        system, user, assistant = [], [], []
        for item in chat:
            if item["role"] == "system":
                system.append(item["content"])
            elif item["role"] == "assistant":
                instruction = "\n".join(system)
                for idx in range(len(user)):
                    instruction += f"\nUSER: {user[idx]}"
                    if idx < len(assistant):
                        instruction += f"\nASSISTANT: {assistant[idx]}"
                instruction += "\nASSISTANT: "
                yield {
                    "instruction": instruction,
                    "response": item["content"],
                    "category": "chat",
                    "skip_counting": True,
                    "skip_prompt_formatting": True,
                }
                assistant.append(item["content"])
            else:
                user.append(item["content"])

        # We'll also yield the complete chat object, to save it as-is.
        yield {"category": "chat", "chat": chat}

        topic_index += 1
        if topic_index >= len(topics):
            topic_index = 0
