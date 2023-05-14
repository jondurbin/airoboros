import sys
from .self_instruct import generate_instructions
from .self_instruct import generate_topics

COMMAND_MAP = {
    "generate-instructions": generate_instructions,
    "generate-topics": generate_topics,
}


def run():
    """Run various commands."""
    args = sys.argv[1:]
    if not args or args[0] not in COMMAND_MAP:
        print(
            f"Please specify one of the supported commands: {', '.join(list(COMMAND_MAP))}"
        )
        sys.exit(1)
    COMMAND_MAP[args[0]](args[1:])


if __name__ == "__main__":
    run()
