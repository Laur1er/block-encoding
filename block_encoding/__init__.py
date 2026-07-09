"""A quantum computing library for block-encoding."""

__version__ = "0.1.0"

from block_encoding.oracles.binary_tree import BinaryAmplitudeTree
from block_encoding.oracles.prepare import build_prepare_oracle

from block_encoding.encode_lcu import encode_LCU
