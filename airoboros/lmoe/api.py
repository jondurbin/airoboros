import argparse
import asyncio
import bitsandbytes as bnb
import copy
import datetime
import fastapi
import glob
import json
import os
import re
import time
import torch
import uuid
import uvicorn
import warnings
from airoboros.lmoe.router import Router
from bitsandbytes.functional import dequantize_4bit
from fastapi import Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from peft import PeftModel
from peft.utils import _get_submodules
from pydantic import BaseModel
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import List, Dict, Any

warnings.filterwarnings("ignore")
MODEL_LOCK = asyncio.Lock()
MODELS = {}
DESCRIPTIONS = {}
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
# Hacky way to handle variations of differently tokenized values...
USER_STOP_TOKENS = [
    torch.tensor([3148, 1001, 29901], device="cuda"),
    torch.tensor([11889, 29901], device="cuda"),
]
ROUTING_PROMPT_TEMPLATE = """A chat.
USER: As an AI assistant, choose the correct function from the list of available functions below, according to the user's request. Your response should be in JSON format.

Input: {instruction}

Available functions:
{functions}
ASSISTANT: """

app = fastapi.FastAPI()


class ChatRequest(BaseModel):
    model: str
    expert: str = None
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
        self.stops = [stop for stop in stops + USER_STOP_TOKENS]

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
                "created": int(time.time()),
                "owned_by": "airoboros",
            }
            for model_id in MODELS
        ],
    }


def route_via_agent(model: Any, request: ChatRequest, stopping_criteria: Any) -> str:
    """Route a request using the LLM with the adapter descriptions."""
    loaded_expert = getattr(model, "__expert__", None)
    if loaded_expert != "function":
        model.set_adapter("function")
        setattr(model, "__expert__", "function")

    # We'll just use the system prompt and last message for the instruction.
    instruction = " ".join(
        [
            request.messages[0]["content"].strip(),
            request.messages[-1]["content"],
        ]
    )
    functions = "\n".join(
        [
            f"{name}:\n  description: {description}"
            for name, description in DESCRIPTIONS.items()
        ]
    )
    prompt = ROUTING_PROMPT_TEMPLATE.format(
        instruction=instruction, functions=functions
    )
    input_ids = MODELS["__tokenizer__"](prompt, return_tensors="pt")["input_ids"].to(
        "cuda"
    )
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=True, enable_mem_efficient=False
    ):
        outputs = model.generate(
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            max_new_tokens=16,
            temperature=0.3,
            top_p=0.8,
            top_k=50,
            use_cache=False,
        )
    response = (
        MODELS["__tokenizer__"]
        .batch_decode(
            outputs.detach().cpu().numpy(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .split("ASSISTANT:")[1]
        .strip()
    )
    response = re.sub(r"[\s\n]*(USER|ASSISTANT):\s*$", "", response, re.DOTALL)
    result = re.search(r'"function":\s*"([^"]+)"', response, re.I)
    if not result:
        result = re.search(
            '"(' + "|".join([re.escape(name) for name in DESCRIPTIONS]) + ')"',
            response,
            re.I,
        )
    if result:
        expert = result.group(1)
        logger.info(f"Agent-based routing selection: {expert}")
        return expert
    return None


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
    request_id = f"cmpl-{uuid.uuid4()}"
    logger.debug(f"Received chat completion request [{request_id}]: {request}")

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
                expected = "assistant"
            else:
                expected = "user"
    prompt = "\n".join(prompt_parts + ["ASSISTANT: "])

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

    # Update our stopping criteria.
    stop_words = request.stop or DEFAULT_STOPS
    stopping_criteria = None
    if stop_words:
        stop_words_ids = [
            MODELS["__tokenizer__"](stop_word, return_tensors="pt")["input_ids"].to(
                "cuda"
            )[0][1:]
            for stop_word in stop_words
        ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

    # Route the request to the appropriate expert (LoRA).
    started_at = datetime.datetime.utcnow()
    model = MODELS[request.model]["model"]
    expert = request.expert
    if expert and expert not in DESCRIPTIONS:
        raise HTTPException(
            status_code=422,
            detail="Invalid expert requested.",
        )
    if not expert:
        if "router" in MODELS[request.model]:
            expert = MODELS[request.model]["router"].route(prompt)
        else:
            expert = route_via_agent(model, request, stopping_criteria)
            if not expert or expert not in DESCRIPTIONS:
                logger.warning("Error performing expert selection, using default")
                expert = "reasoning"

    # Load the adapter.
    model = MODELS[request.model]["model"]
    loaded_expert = getattr(model, "__expert__", None)
    if loaded_expert != expert:
        model.set_adapter(expert)
        setattr(model, "__expert__", expert)
    routing_duration = (datetime.datetime.utcnow() - started_at).total_seconds()

    # Generate the response.
    started_at = datetime.datetime.utcnow()
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=True, enable_mem_efficient=False
    ):
        outputs = model.generate(
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            max_new_tokens=max_tokens,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            use_cache=False,
        )
    response = (
        MODELS["__tokenizer__"]
        .batch_decode(
            outputs.detach().cpu().numpy(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .split("ASSISTANT:")[-1]
        .strip()
    )
    response = re.sub(r"[\s\n]*(USER|ASSISTANT):\s*$", "", response, re.DOTALL)
    duration = (datetime.datetime.utcnow() - started_at).total_seconds()
    logger.debug(f"Response: {response}")
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


def dequantize_model(model, save_path):
    """Dequantize a model.  This is a bit odd, but evidently performance is better
    when you first quantize a model, then dequantize, then merge lora adapters.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, bnb.nn.Linear4bit):
                continue
            quant_state = copy.deepcopy(module.weight.quant_state)
            quant_state[2] = torch.bfloat16
            weights = dequantize_4bit(
                module.weight.data, quant_state=quant_state, quant_type="nf4"
            ).to(torch.bfloat16)
            new_module = torch.nn.Linear(
                module.in_features, module.out_features, bias=None, dtype=torch.bfloat16
            )
            new_module.weight = torch.nn.Parameter(weights)
            new_module.to(device="cuda", dtype=torch.bfloat16)
            parent, target, target_name = _get_submodules(model, name)
            setattr(parent, target_name, new_module)
        model.is_loaded_in_4bit = False
        model.save_pretrained(save_path)
        with open(os.path.join(save_path, "config.json")) as infile:
            config = json.loads(infile.read())
        config.pop("quantization_config", None)
        config.pop("pretraining_tp", None)
        with open(os.path.join(save_path, "config.json"), "w") as outfile:
            outfile.write(json.dumps(config, indent=2))
        logger.debug(f"Saved dequantized model to {save_path}")
        return model


def initialize_lmoe(base, lmoe, cache_dir):
    """Load the model, quantizing first, then dequantizing, then merging adapters."""
    base_name = os.path.basename(base)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    dequant_path = os.path.join(cache_dir, f"{base_name}.dequantized")
    base_model = None
    if os.path.isdir(dequant_path):
        base_model = AutoModelForCausalLM.from_pretrained(
            dequant_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_model = dequantize_model(
            AutoModelForCausalLM.from_pretrained(
                os.path.abspath(base),
                load_in_4bit=True,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                device_map="auto",
            ),
            dequant_path,
        )
    return PeftModel.from_pretrained(
        base_model.to_bettertransformer().eval(),
        os.path.abspath(os.path.join(lmoe, "adapters", "function")),
        adapter_name="function",
    )


def main():
    parser = argparse.ArgumentParser(
        description="airoboros LMoE API server, somewhat similar to OpenAI API.",
    )
    parser.add_argument("-i", "--host", type=str, default="127.0.0.1", help="host name")
    parser.add_argument("-p", "--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "-k",
        "--router-k",
        type=int,
        default=20,
        help="k, when doing faiss approximate knn search to select expert",
    )
    parser.add_argument(
        "-e",
        "--router-model",
        type=str,
        default="thenlper/gte-small",
        help="model to use for embeddings in expert router",
    )
    parser.add_argument(
        "-a",
        "--agent-router",
        action="store_true",
        help="use the function/agent adapter to route requests instead of embeddings",
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
    parser.add_argument(
        "-c",
        "--cache-dir",
        type=str,
        help="cache directory to save dequantized model to",
        default=".cache",
    )
    args = parser.parse_args()

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

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
            "model": initialize_lmoe(base, lmoe, args.cache_dir),
        }
        if not args.agent_router:
            MODELS[base_name]["router"] = Router(
                model_name_or_path=args.router_model,
                input_paths=routing_paths,
                max_samples=args.router_max_samples,
                k=args.router_k,
            )
        logger.info(
            f"Loading adapters for {base_name} from {lmoe}: activating all thrusters..."
        )
        for path in tqdm(glob.glob(os.path.join(lmoe, "adapters", "*"))):
            name = os.path.basename(str(path))
            if name == "function":
                continue
            MODELS[base_name]["model"].load_adapter(str(path), name)
            with open(os.path.join(str(path), "description.txt")) as infile:
                DESCRIPTIONS[name] = infile.read().strip()

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
