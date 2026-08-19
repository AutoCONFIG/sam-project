"""子模型 B: 文本编码器 (TextTransformer + resizer)

来源: multiplex predictor 的 detector.backbone.language_backbone
(VETextEncoder)。tokenizer (BPE) 不是 nn.Module, 不导出 — 输入直接是
token ids, 端侧用 Python/JS 实现相同 tokenization (词表在
sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz)。
"""

from typing import Dict, List, Tuple

import torch

from .utils import ExportOptions, export_component
from .wrappers import TextEncoderWrapper

NAME = "text_encoder"

_VOCAB_SIZE = 49408  # SimpleTokenizer 词表大小 (bpe_simple_vocab_16e6)

INPUT_NAMES: List[str] = ["token_ids"]
OUTPUT_NAMES: List[str] = ["text_memory", "text_mask", "text_embeds"]


def build_wrapper(predictor, resolution: int) -> TextEncoderWrapper:
    text_encoder = predictor.model.detector.backbone.language_backbone
    return TextEncoderWrapper(text_encoder)


def dummy_inputs(resolution: int, device) -> Tuple[torch.Tensor, ...]:
    wrapper_context_length = 32  # VETextEncoder context_length 固定 32
    token_ids = torch.randint(
        0, _VOCAB_SIZE, (1, wrapper_context_length), dtype=torch.int64, device=device
    )
    return (token_ids,)


def dynamic_axes() -> Dict:
    return {
        "token_ids": {0: "batch"},
        "text_memory": {1: "batch"},
        "text_mask": {0: "batch"},
        "text_embeds": {1: "batch"},
    }


def export(predictor, options: ExportOptions, output_dir):
    wrapper = build_wrapper(predictor, options.resolution)
    device = next(wrapper.parameters()).device
    inputs = dummy_inputs(options.resolution, device)
    return export_component(
        NAME, wrapper, inputs, INPUT_NAMES, OUTPUT_NAMES, options, output_dir,
        dynamic_axes=dynamic_axes(),
    )
