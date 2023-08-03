import re
import json
import uuid
inputs = [json.loads(line) for line in open("instructions.jsonl").readlines()]

def split_response(instruction, response):
    if '</s>' not in response:
        return [
            {
                "from": "human",
                "value": instruction,
            },
            {
                "from": "gpt",
                "value": response,
            },
        ]
    parts = response.split('</s>')
    user = [instruction]
    assistant = []
    for idx in range(len(parts)):
        part = parts[idx]
        if idx == 0:
            assistant.append(part)
            continue
        match = re.match(r'^\s*USER:(.*?)ASSISTANT:(.*)\s*$', part, re.DOTALL)
        if not match:
            return None
        user.append(match.group(1).strip())
        assistant.append(match.group(2).strip())
    conv = []
    for idx in range(len(user)):
        conv.append({
            "from": "human",
            "value": user[idx],
        })
        conv.append({
            "from": "gpt",
            "value": assistant[idx]
        })
    return conv

conversations = []
for row in inputs:
    conversation = split_response(row['instruction'], row['response'])
    if not conversation:
        print("Bad format, skipping...")
        continue
    conversations.append({
        "id": str(uuid.uuid4()),
        "conversations": conversation,
    })
with open("as_conversations.json", "w") as outfile:
    outfile.write(json.dumps(conversations, indent=2))
