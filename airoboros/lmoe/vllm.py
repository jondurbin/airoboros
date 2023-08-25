# Modified from original vLLM adapter:
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
import argparse
import asyncio
import glob
import json
import os
import time
import torch
from http import HTTPStatus
from typing import AsyncGenerator, Optional
import fastapi
import uvicorn
import vllm.entrypoints.openai.api_server as api_server_mod
from fastapi import BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from peft import PeftConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import (
    create_error_response,
    check_model,
    check_length,
)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
    ModelCard,
    ModelList,
    ModelPermission,
)
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid
from airoboros.lmoe.router import Router
from airoboros.lmoe.lora import lora_merge_unmerge_state_dict

TIMEOUT_KEEP_ALIVE = 5  # seconds
MODEL_LOCK = asyncio.Lock()
ROLE_MAP = {
    "user": "USER",
    "assistant": "ASSISTANT",
}

served_model = None
tokenizer = None
app = fastapi.FastAPI()


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model, root=served_model, permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


async def complete_request(raw_request, request):
    """Complete a chat request, which is wrapped by asyncio lock."""

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
            return create_error_response(
                HTTPStatus.BAD_REQUEST, f"Invalid role found: {message['role']}"
            )
        elif message["role"] != expected:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "Invalid messages structure, expected system -> [user assistant]* user",
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
    logger.debug(f"Prompt:\n{prompt}")

    # Route the request to the appropriate expert (LoRA).
    expert = router.route(prompt)
    loaded_expert = getattr(engine, "__expert__", None)
    if loaded_expert != expert:
        if loaded_expert is not None:
            lora_merge_unmerge_state_dict(
                engine.engine,
                adapters[loaded_expert],
                adapter_configs[loaded_expert],
                merge=False,
            )
        lora_merge_unmerge_state_dict(
            engine.engine, adapters[expert], adapter_configs[expert], merge=True
        )
        setattr(engine, "__expert__", expert)

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "logit_bias is not currently supported"
        )

    token_ids, error_check_ret = await check_length(request, prompt)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.time())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt, sampling_params, request_id)

    async def abort_request() -> None:
        await engine.abort(request_id)

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(content=text),
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id, choices=[choice_data], model=model_name
            )
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]) :]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(
            completion_stream_generator(),
            media_type="text/event-stream",
            background=background_tasks,
        )

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST, "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=ChatMessage(role="assistant", content=output.text.strip()),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            fake_stream_generator(), media_type="text/event-stream"
        )

    return response


@app.post("/v1/chat/completions")
async def create_chat_completion(raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    request = ChatCompletionRequest(**await raw_request.json())
    async with MODEL_LOCK:
        return await complete_request(raw_request, request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
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
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not "
        "specified, the model name will be the same as "
        "the huggingface name.",
    )
    parser.add_argument(
        "--router-max-samples",
        type=int,
        default=1000,
        help="Maximum number of training samples per expert "
        "to use when building router index.",
    )
    parser.add_argument(
        "--router-k",
        type=int,
        default=25,
        help="k, when doing faiss approximate knn search to select expert",
    )
    parser.add_argument(
        "--lmoe",
        type=str,
        required=True,
        help="Path to LMoE directory with adapters and data",
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.get_max_model_len()

    # Setup our router, and load all of the adapters so they
    # are ready to swap in/out.
    routing_paths = [
        str(p) for p in glob.glob(os.path.join(args.lmoe, "routing_data", "*.jsonl"))
    ]
    router = Router(
        input_paths=routing_paths, max_samples=args.router_max_samples, k=args.router_k
    )
    adapters = {}
    adapter_configs = {}
    for directory in glob.glob(os.path.join(args.lmoe, "adapters", "*")):
        adapters[str(directory).split("/")[-1]] = torch.load(
            os.path.join(str(directory), "adapter_model.bin"), map_location="cuda:0"
        )
        adapter_configs[str(directory).split("/")[-1]] = PeftConfig.from_json_file(
            os.path.join(str(directory), "adapter_config.json")
        )

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=engine_args.trust_remote_code,
    )
    api_server_mod.tokenizer = tokenizer
    api_server_mod.served_model = served_model
    api_server_mod.max_model_len = max_model_len
    api_server_mod.engine = engine
    api_server_mod.engine_args = engine_args
    api_server_mod.engine_model_config = engine_model_config
    api_server_mod.app = app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
