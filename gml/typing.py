"""Type definitions."""

import torch as _th
from typing import Iterable, TypeAlias
Device: TypeAlias = _th.device | str
EdgeList: TypeAlias = Iterable[tuple[int, int]]


# Re-export types.
from torch import Tensor
from typing import Any, Optional
