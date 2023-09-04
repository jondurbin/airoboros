import re
import os
import glob
import hashlib
import json
import random
from difflib import get_close_matches
from loguru import logger
from airoboros.instructors.character import generate as generate_character


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
Pay close attention to the era/time when {name} exists/existed, if specified, and keep that in mind when generating responses as {name}; for example someone from the 1800s would have no idea what a cell phone is.
Don't be too aggreable - act as the character, hold to their beliefs, mood, etc., don't just abruptly change your entire philosophy.
Tune down your normal sunny disposition - not every conversation is cheery and fun.
If the subject suddenly changes, it's appropriate to hesitate, since an abrupt change in real conversations would be strange, but if continually pressed you will comply.
Avoid making toasts, cheering, etc., or other type of verbiage that could indicate an attempt to wrap up the conversation.
Never start your response with an acknowedgement of the input, such as "Indeed, the idea of [..] is intriguing."
Never start any sentence with "Ah, ".
Never start by complementing or acknowleding the message, e.g. "That\'s an interesting ..."
Never start any response with "Indeed...".
Never speak in the third person or make a reference to your character's name in the response.
It is clear that you are the one performing the action/speaking, so you must never reference your character's name.
Before generating any output, read all of the input and previous output carefully, then ensure all of your output contains unique phrases, actions, verbs, adjectives, etc. that haven't previously been used.
Don't be repetitive.  Make sure you never repeat questions, or points already made.
Carefully read all of the the previous content and be sure NEVER include any repeated actions, phrases, thoughts, etc. that have already appeared.
Remember - DO NOT REPEAT ACTIONS OR PHRASES
Select from a wide collection of verbs, adjectives, adverbs, etc. and be as diverse as possible in the output, using new words not previously seen in either outputs or inputs.
Try to keep the response to 150 words or less.
{flesch}
"""
NON_USER_RULES = """
All characters should speak in roughly equal quantities, however be sure to include {user_name} slightly more often.
Give detailed, immersive, colorful, insightful, and intelligent responses, and be sure to re-read the system prompt and follow the guidance and chat setting provided therein.
"""
FORMATTING = """
Actions the character performs must be differentiated from spoken words and general narration:
 - example: {action_delim}slowly lifting her gaze{action_delim}
Keep the actions fairly succint and lowercase, and combine any immediately adjacent actions.
Characters must avoid repeating actions.
Actions will not include any references to the actor or include "I [some action]".
For example, instead of {action_delim}I raise an eyebrow{action_delim}, the action would be {action_delim}raises an eyebrow{action_delim}.
Never start the action with "I ..."
"""
QUOTING = """
Words spoken by the character must be differentiated from actions and general narration, and should be quoted.
 - example: "That really surprises me!"
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
START = [
    "Start the conversation.",
    "Begin the dialogue.",
    "Let's initiate our discussion.",
    "Kick off the chat.",
    "Open the lines of communication.",
    "Let's dive into the topic.",
    "Commence our talk.",
    "Let's break the ice.",
    "Time to start talking.",
    "Fire up the conversation.",
    "Let's get the ball rolling.",
    "Jump right in.",
    "Embark on our discussion.",
    "Let's get things started.",
    "Time to open up.",
    "Get the conversation flowing.",
    "Let's touch base on this.",
    "Time to engage in dialogue.",
    "Begin the discourse.",
    "Let's get into it.",
    "Start the chatter.",
    "Initiate the exchange.",
    "Begin the chat.",
    "Let's set the stage with a conversation.",
    "Dive into our chat.",
    "Let's start discussing.",
]


def parse_response(response, current_name, user_name, names, action_delim):
    """ "Extract the next character from the response."""
    # We'll also take this opportunity to remove any disclaimers.
    response = response.split("REMINDER:")[0].strip()
    response = response.split("RULES:")[0].strip()
    match = re.search(r"(NEXT:\s*([^\n]+))", response)
    name = user_name
    if not match:
        logger.warning(f"Didn't generate NEXT target: {response}")
        if current_name == user_name:
            name = random.choice(list(set(names) - set([user_name])))
    else:
        response = response.replace(match.group(1), "").strip()
        name = match.group(2).strip().replace('"', "")
    if response.startswith(f"{current_name}:"):
        response = response[len(current_name) + 1 :].lstrip()
    response = response.replace("“", '"').replace("”", '"')

    # Clean up any hallucinated responses on behalf of other names.
    other_names = set(names) - set([current_name])
    if current_name != user_name:
        other_names.add(user_name)
    other_names_re = (
        "\n(" + "|".join([str(re.escape(name)) for name in other_names]) + "):"
    )
    response = re.split(other_names_re, response)[0]

    # Cleanup stray action delimiters and other garbage output.
    if action_delim:
        action = re.escape(action_delim)
        response = re.sub(
            f'({action})([\.,\s-]*){action}\s*"',
            r'\1 "',
            response,
        )
        response = re.sub(
            f'"([\W]{0,2})"({action})',
            r'"\1\2',
            response,
        )
        response = re.sub(
            f'({action})"([\W]{0,2})"',
            r'\1\2"',
            response,
        )
        response = re.sub(f"{action}\(|\){action}", action_delim, response)
        response = re.sub(f"[,\.]{action}", action_delim, response)
        response = re.sub(f"({action})[,\.](\s)", r"\1\2", response)
        response = re.sub(r" +", " ", response)

    # Handle any misspellings or 's, etc. in case the name doesn't match.
    if name not in names:
        matches = get_close_matches(name, list(set(names) | set([user_name])))
        if not matches:
            name = random.choice(names)
        else:
            name = matches[0]
    if name not in list(names) + [user_name]:
        if current_name.startswith(user_name):
            name = random.choice(names)
        else:
            name = user_name
    return response, name


async def generate_cards(instructor):
    """Load character cards to use for the various characters during chat."""
    card_config = instructor.instructors["character"]

    # Load the existing character cards.
    cards = []
    cards_dir = card_config.get("output_dir", "characters")
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
        async for item in generate_character(instructor, skip=skip):
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
            logger.success(f"Generated character card {filename}")
            if len(cards) >= card_count:
                break
    instructor.instructor_counts["character"] = len(cards)
    return cards


async def generate_setting(instructor, user_card, characters, topic, **api_params):
    """Generate a setting to use for the RP session."""
    path = instructor.instructors["rp"].get("setting_prompt_path") or "rp_setting.txt"
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
    """Generate the first message."""
    user_name = user_card["name"]
    messages = {name: [] for name in set(list(characters) + [user_name])}
    flesch = (
        instructor.instructors.get("rp", {}).get("flesch") or instructor.default_flesch
    )
    action_delim = random.choice(["*", "~", None])
    first_name = None
    all_names = list(characters) + [user_name]
    setting = await generate_setting(
        instructor, user_card, characters, topic, **api_params
    )
    character_block = "\n\n".join(
        [
            f'{name}: {card["description"].strip()}'
            for name, card in {**characters, **{user_name: user_card}}.items()
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
                    f"Setting for the chat:\n{setting}\nEnd of setting.",
                ]
            ),
        }
    ]
    if action_delim:
        training[0][
            "content"
        ] += f"\nActions should  be surrounded by {action_delim}, e.g. {action_delim}slowly turns his gaze towards the lamp{action_delim}"
    logger.success(f"Generated the system prompt:\n{training[0]['content']}")
    formatting = QUOTING
    if action_delim:
        formatting = FORMATTING.format(action_delim=action_delim) + QUOTING
    for name in all_names:
        if not first_name:
            first_name = name
        others = list(set(all_names) - set([name]))

        # Create the OpenAI messages to use for generating responses, which require some extra rules.
        hint = (
            characters[name]["stay_in_character"]
            if name != user_name
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
                            name=name,
                        ),
                        formatting,
                        (
                            ""
                            if name == user_name
                            else NON_USER_RULES.format(user_name=user_name)
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
    example_message, next_name = parse_response(
        example_message, first_name, user_card["name"], all_names, action_delim
    )
    if not example_message or not example_message.strip():
        raise RuntimeError("Failed to generate example message")

    # Add a random user instruction to initiate the chat as the actual instruction.
    training.append(
        {
            "role": "user",
            "content": random.choice(START),
        }
    )

    # Add the first response.
    training.append(
        {"role": "assistant", "content": f"{first_name}: {example_message}"}
    )

    # Update all of the other characters' messages with the first response.
    logger.success(
        f"Generated the opening [from: {first_name}, next: {next_name}]:\n{example_message}"
    )
    for name in set(all_names) - set([first_name]):
        messages[name].append(
            {
                "role": "user",
                "content": f"{first_name}: {example_message}",
            }
        )
    return training, messages, first_name, next_name, action_delim


async def generate_rp(instructor, cards, topic, **api_params):
    """Generate one new session using the provided cards/topic."""

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
    user_name = user_card["name"]
    current_name = next_name
    all_names = list(characters) + [user_name]
    flesch = (
        instructor.instructors.get("rp", {}).get("flesch") or instructor.default_flesch
    )
    target_turns = instructor.instructors["rp"].get("turn_count") or 50
    topics = instructor.get_instructor_topics(instructor.instructors["rp"])
    rerolls = 0
    while True:
        others = list(set(all_names) - set([current_name]))

        # Continue the conversation, occasionally injecting random turns and topic changes.
        next_turn = random.choice(CONV_TURNS)
        if "{topic}" in next_turn:
            change_topic = random.choice(topics)
            while topic == change_topic:
                change_topic = random.choice(topics)
            next_turn = next_turn.format(topic=json.dumps(change_topic))

        # Re-iterate the rules.
        messages[current_name][-1]["content"] += "\n" + "\n".join(
            [
                "RULES:\nRemember, you must always stay in character.",
                RULES.format(flesch=flesch, name=current_name),
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

        # Remove the re-iterated rules from the prompt to reduce token usage for the next iteration.
        messages[current_name][-1]["content"] = (
            messages[current_name][-1]["content"].split("RULES:")[0].strip()
        )
        if not response or not response.strip():
            if rerolls > 3:
                logger.error("Max rerolls, stopping generation.")
                break
            logger.warning("No response, rerolling!")
            rerolls += 1
            continue

        # Clean up and extract next name.
        response, next_name = parse_response(
            response, current_name, user_card["name"], all_names, action_delim
        )
        if not response or not response.strip():
            if rerolls > 3:
                logger.error("Max rerolls, stopping generation.")
                break
            logger.warning("No continuation resonse, rerolling!")
            rerolls += 1
            continue

        # Update the current character's message history.
        messages[current_name].append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # Update training data.
        prefix = f"{current_name}: " if current_name != user_name else f"{user_name}: "
        if current_name != user_name and training[-1]["role"] == "assistant":
            training[-1]["content"] += f"\n\n{prefix}{response}"
        else:
            training.append(
                {
                    "role": "assistant" if current_name != user_name else "user",
                    "content": f"{prefix}{response}",
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
            current_name == user_name and len(training) >= target_turns - 1
        ):
            logger.success(f"Reached {len(training)}, finished.")
            break
    return training


async def generate(instructor, **kwargs):
    """Generator for roleplay training data.  Yes, this is slightly confusing, because
    we already have a 'roleplay' instructor, but that one is meant for answering normally,
    but influenced by a particular style or person.  Here, this is meant for interactive
    RP chat with emotes and the like.
    """
    config = instructor.instructors.get("rp", {})
    if not config:
        return
    card_config = instructor.instructors.get("character", {})
    if not card_config:
        return
    target_count = instructor.instructors["rp"].get("count")
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

    # Start generating.
    while instructor.instructor_counts["rp"] < target_count:
        # Select a random number of characters.
        count_options = [2, 3, 4]
        count_weights = [0.85, 0.12, 0.03]
        card_count = random.choices(count_options, count_weights)[0]
        rp = await generate_rp(
            instructor,
            random.sample(cards, card_count),
            topics[topic_index],
            **api_params,
        )
        if not rp:
            continue

        # We'll convert each round into a row of training data, i.e.:
        # instruction 0 = system + user, response 0 = assistant response 0
        # instruction 1 = system + user + assistant + user, response 1 = assistant response 1
        # This way all of our existing training scripts should work without any changes.
        system, user, assistant = [], [], []
        counted = False
        for item in rp:
            if item["role"] == "system":
                system.append(item["content"])
            elif item["role"] == "assistant":
                instruction = "\n".join(system)
                for idx in range(len(user)):
                    instruction += f"\n{user[idx]}"
                    if idx < len(assistant):
                        instruction += f"\n{assistant[idx]}"
                instruction = instruction.strip() + "\n"
                yield {
                    "instruction": instruction,
                    "response": item["content"],
                    "category": "rp",
                    "skip_counting": False if not counted else True,
                    "skip_prompt_formatting": True,
                }
                counted = True
                assistant.append(item["content"])
            else:
                user.append(item["content"])

        topic_index += 1
        if topic_index >= len(topics):
            topic_index = 0
