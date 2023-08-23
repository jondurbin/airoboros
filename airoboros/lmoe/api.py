import argparse
import asyncio
import datetime
import fastapi
import glob
import os
import time
import torch
import uuid
import uvicorn
import warnings
from airoboros.lmoe.router import Router
from fastapi import Request, HTTPException
from loguru import logger
from peft import PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import List, Dict

warnings.filterwarnings("ignore")
MODEL_LOCK = asyncio.Lock()
MODELS = {}
ROLE_MAP = {
    "user": "USER",
    "assistant": "ASSISTANT",
}
DEFAULT_STOPS = [
    "USER:",
    "ASSISTANT:",
    "### Instruction",
    "### Response",
    # These are often used as refusals, warnings, etc, but may also remove useful info.
    # "\nRemember,"
    # "\nPlease note,"
]

app = fastapi.FastAPI()


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.5
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop: List[str] = DEFAULT_STOPS
    max_tokens: int = None


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False


@app.get("/v1/models")
async def list_models():
    """Show available models."""
    # TODO: use HF to get this info.
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.now()),
                "owned_by": "airoboros",
            }
            for model_id in MODELS
        ],
    }


def complete_request(request):
    """Sync method to complete a request, to make sure we aren't message with model/LoRAs concurrently."""
    if any(
        [
            (getattr(request, key, None) or 0) < 0
            for key in [
                "temperature",
                "repetition_penalty",
                "top_p",
                "top_k",
                "max_tokens",
            ]
        ]
    ):
        raise HTTPException(status_code=422, detail="Bad request (params < 0)")

    # Really, really basic validation.
    if not request.model or request.model not in MODELS:
        raise HTTPException(status_code=404, detail="Model not available.")

    # Make sure we have a system prompt.
    if request.messages[0]["role"] != "system":
        request.messages = [{"role": "system", "content": "A chat."}] + request.messages
    logger.debug(f"Received chat completion request: {request}")

    # Build the prompt, with a bit more (very basic) validation.
    prompt_parts = []
    expected = "system"
    for message in request.messages:
        if message["role"] == "system":
            prompt_parts.append(message["content"])
            expected = "user"
        elif message["role"] not in ROLE_MAP:
            raise HTTPException(
                status_code=422, detail="Invalid role found: {message['role']}"
            )
        elif message["role"] != expected:
            raise HTTPException(
                status_code=422,
                detail="Invalid messages structure, expected system -> [user assistant]* user",
            )
        else:
            prompt_parts.append(
                f"{ROLE_MAP[message['role']]}: {message['content'].strip()}"
            )
            if message["role"] == "user":
                expected == "assistant"
            else:
                expected == "user"
    prompt = " ".join(prompt_parts + ["ASSISTANT: "])

    # Validate the length of the input.
    input_ids = MODELS["__tokenizer__"](prompt, return_tensors="pt")["input_ids"].to(
        "cuda"
    )
    max_len = MODELS[request.model]["config"].max_position_embeddings
    max_tokens = request.max_tokens or max_len - len(input_ids[0]) - 1
    if len(input_ids[0]) + max_tokens > max_len:
        raise HTTPException(
            status_code=422,
            detail="Prompt length + max_tokens exceeds max model length.",
        )

    # Route the request to the appropriate expert (LoRA).
    started_at = datetime.datetime.utcnow()
    expert = MODELS[request.model]["router"].route(prompt)
    model = MODELS[request.model]["model"]
    loaded_expert = getattr(model, "__expert__", None)
    if loaded_expert != expert:
        model.set_adapter(expert)
        setattr(model, "__expert__", expert)
    routing_duration = (datetime.datetime.utcnow() - started_at).total_seconds()

    # Update our stopping criteria.
    stop_words = request.stop or DEFAULT_STOPS
    stopping_criteria = None
    if stop_words:
        stop_words_ids = [
            MODELS["__tokenizer__"](stop_word, return_tensors="pt")["input_ids"]
            .to("cuda")
            .squeeze()
            for stop_word in stop_words
        ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

    # Generate the response.
    started_at = datetime.datetime.utcnow()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            max_new_tokens=max_tokens,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
    response = (
        MODELS["__tokenizer__"]
        .batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        .split("ASSISTANT:")[1]
        .strip()
    )
    duration = (datetime.datetime.utcnow() - started_at).total_seconds()
    request_id = f"cmpl-{uuid.uuid4()}"
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "duration": duration,
        "routing_duration": routing_duration,
        "model": request.model,
        "expert": expert,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.strip(),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(input_ids[0]),
            "completion_tokens": len(outputs[0]),
            "total_tokens": len(input_ids[0]) + len(outputs[0]),
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    """Simulate the OpenAI /v1/chat/completions endpoint.

    NOTE: Parameters supported in request include:
        - model: str, must be loaded from CLI args.
        - messages: list[dict[str, str]]
        - temperature: float
        - repetition_penalty: float
        - top_p: float
        - top_k: int
        - stop: list[str]
        - max_tokens: int

    Example request:
    curl -s -XPOST http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
      "model": "airoboros-lmoe-7b-2.1",
      "messages": [
        {
          "role": "system",
          "content": "A chat.",
        },
        {
          "role": "user",
          "content": "Write a poem about Ouroboros."
        }
      ]
    }'
    """
    request = ChatRequest(**await raw_request.json())
    async with MODEL_LOCK:
        return complete_request(request)


def main():
    parser = argparse.ArgumentParser(
        description="airoboros LMoE API server, somewhat similar to OpenAI API.",
    )
    parser.add_argument("-i", "--host", type=str, default="127.0.0.1", help="host name")
    parser.add_argument("-p", "--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "-k",
        "--router-k",
        type=int,
        default=20,
        help="k, when doing faiss approximate knn search to select expert",
    )
    parser.add_argument(
        "-s",
        "--router-max-samples",
        type=int,
        default=1000,
        help="number of samples to include in router faiss indices per expert",
    )
    parser.add_argument(
        "-b",
        "--base-model",
        type=str,
        help="base model(s) to load",
        nargs="+",
    )
    parser.add_argument(
        "-l",
        "--lmoe",
        type=str,
        help="lmoe adapter package to load",
        nargs="+",
    )
    args = parser.parse_args()

    # Load all of the models and the corresponding adapters.
    for base, lmoe in zip(args.base_model, args.lmoe):
        base_name = os.path.basename(base)
        routing_paths = [
            str(p) for p in glob.glob(os.path.join(lmoe, "routing_data", "*.jsonl"))
        ]
        logger.info(
            f"Initializing base model {base_name}: it'll be a while, go have a snack"
        )
        if "__tokenizer__" not in MODELS:
            MODELS["__tokenizer__"] = AutoTokenizer.from_pretrained(
                os.path.abspath(base)
            )
        MODELS[base_name] = {
            "config": AutoConfig.from_pretrained(base),
            "model": PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(
                    os.path.abspath(base), device_map="auto", torch_dtype=torch.float16
                ),
                os.path.abspath(os.path.join(lmoe, "adapters", "general")),
                adapter_name="general",
            ),
            "router": Router(
                input_paths=routing_paths,
                max_samples=args.router_max_samples,
                k=args.router_k,
            ),
        }
        logger.info(
            f"Loading adapters for {base_name} from {lmoe}: this too is slow..."
        )
        for path in tqdm(glob.glob(os.path.join(lmoe, "adapters", "*"))):
            name = os.path.basename(str(path))
            if name == "general":
                continue
            MODELS[base_name]["model"].load_adapter(str(path), name)

    # Start the API server.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
