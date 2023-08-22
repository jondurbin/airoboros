# Adapted from: https://github.com/vllm-project/vllm/issues/182#issuecomment-1641486990
#
# Load original llama to vllm with llm = LLM("llama-7b") ...
# Load lora states dict lora_state_dict = torch.load("lora_states.pt")['module'].
# Merge lora states to llm do lora_merge_unmerge_state_dict(llm.llm_engine, lora_state_dict, merge=True)
# Do whatever inference job with llm ...
# To unmerge and obtain the original llama, run lora_merge_unmerge_state_dict(llm.llm_engine, lora_state_dict, merge=False)


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def lora_reassign_weights(
    model, state_dict, r, lora_alpha, fan_in_fan_out=False, merge=True
):
    is_merged = getattr(model, "is_merged", False)
    assert (
        is_merged != merge
    ), f"{is_merged} != {merge}: if is_merged, then must be unmerge; if not is_merged, then must merge"
    named_params = [(n, p) for n, p in model.named_parameters()]
    scaling = lora_alpha / r
    state_dict = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
    replaced = set()
    merged_names = {
        # these are projector weights that got combined into single matrix in vllm
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    non_merged_names = ["o_proj", "down_proj"]
    for name, param in named_params:
        param.requires_grad = False
        if "_proj.weight" not in name:
            continue
        for wn, wn_series in merged_names.items():
            if name.endswith(f"{wn}.weight"):
                for stride_id, att_weight_name in enumerate(wn_series):
                    lora_a = name.replace(
                        f"{wn}.weight", f"{att_weight_name}.lora_A.weight"
                    )
                    lora_b = name.replace(
                        f"{wn}.weight", f"{att_weight_name}.lora_B.weight"
                    )
                    shard_size = param.shape[0] // len(wn_series)
                    if lora_a in state_dict:
                        assert lora_b in state_dict, f"{lora_b} not in state_dict"
                        assert (
                            state_dict[lora_b].shape[1] == r
                        ), f"{r=} != {state_dict[lora_b].shape}"
                        matrix = (
                            transpose(
                                state_dict[lora_b] @ state_dict[lora_a], fan_in_fan_out
                            )
                            * scaling
                        )
                        assert (
                            param.data[
                                shard_size * stride_id : shard_size * (stride_id + 1)
                            ].shape
                            == matrix.shape
                        )
                        if merge:
                            param.data[
                                shard_size * stride_id : shard_size * (stride_id + 1)
                            ] += matrix
                        else:
                            param.data[
                                shard_size * stride_id : shard_size * (stride_id + 1)
                            ] -= matrix
                        replaced.add(lora_a)
                        replaced.add(lora_b)
        for wn in non_merged_names:
            if name.endswith(f"{wn}.weight"):
                lora_a = name.replace(f"{wn}.weight", f"{wn}.lora_A.weight")
                lora_b = name.replace(f"{wn}.weight", f"{wn}.lora_B.weight")
                if lora_a in state_dict:
                    assert lora_b in state_dict
                    matrix = (
                        transpose(
                            state_dict[lora_b] @ state_dict[lora_a], fan_in_fan_out
                        )
                        * scaling
                    )
                    assert (
                        param.data.shape == matrix.shape
                    ), f"invalid shape: {name} {param.data.shape} != {matrix.shape}"
                    if merge:
                        param.data += matrix
                    else:
                        param.data -= matrix
                    replaced.add(lora_a)
                    replaced.add(lora_b)
    no_replaced = [k for k in state_dict.keys() if k not in replaced]
    assert (
        len(no_replaced) == 0
    ), f"some lora states not loaded, check again!: {no_replaced}"
    model.is_merged = merge


def lora_merge_unmerge_state_dict(engine, state_dict, peft_config, merge=True):
    # merge lora states to weights
    for worker in engine.workers:
        lora_reassign_weights(
            worker.model,
            state_dict,
            r=peft_config["r"],
            lora_alpha=peft_config["lora_alpha"],
            fan_in_fan_out=peft_config["fan_in_fan_out"],
            merge=merge,
        )
