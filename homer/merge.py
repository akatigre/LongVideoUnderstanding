import torch

class MergeManager:
    def __init__(
        self,
        max_chunk_len=2048,
        max_initial_chunk_len=-1,
        reduction_mode="power_max_last_calibrated",
        layers_warmup=12,
        target_len=2048,
        bias_path=None,
        visualize=False,
    ):
        # Values updated per experiment
        self.max_chunk_len = max_chunk_len
        self.max_initial_chunk_len = max_initial_chunk_len
        self.reduction_mode = reduction_mode
        self.layers_warmup = layers_warmup
        self.target_len = target_len
        self.visualize = visualize
        self.bias = (
            torch.load(bias_path, map_location="cpu") if bias_path is not None else None
        )

        # Values updated per sample
        self.prefix_len = None
        self.suffix_len = None
        self.eff_max_chunk_len = None
        self.layers_per_chunk = None
        self.layers_leftover = None

        # Values updated per layer
        self.layer_reduction_info = None
        self.layer_reduction_results = None

    def set_sample_info(
        self,
        prefix_len,
        suffix_len,
        eff_max_chunk_len,
        layers_per_chunk,
        layers_leftover,
    ):
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.eff_max_chunk_len = eff_max_chunk_len
        self.layers_per_chunk = layers_per_chunk
        self.layers_leftover = layers_leftover

    def set_layer_reduction_info(
        self,
        num_tokens_to_reduce=0,
        reduction_mask=None,
        position_ids=None,
    ):
        if num_tokens_to_reduce == 0:
            self.layer_reduction_info = None
        else:
            self.layer_reduction_info = {
                "num_tokens_to_reduce": num_tokens_to_reduce,
                "reduction_mask": reduction_mask,
                "position_ids": position_ids,
            }

    def set_layer_reduction_results(
        self,
        position_ids=None,
        prune_mask=None,
        significance_weights=None,
    ):
        self.layer_reduction_results = {
            "position_ids": position_ids,
            "prune_mask": prune_mask,
            "significance_weights": significance_weights,
        }
