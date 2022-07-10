"""Utilities that support other modules."""

import torch

from .typing import Any, Tensor

class Namespace(object):
    """A simple container that allows accessing values via attributes.

    When storing a tensor, the first dimension must match the predefined
    dimension for this container.

    # Examples

    ```
    >>> import torch
    >>> from gml.utils import Namespace
    >>>
    >>> data = Namespace(2)
    >>> data.foo = 123
    >>> data.bar = torch.ones(2, 3)
    >>> data.bar
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    >>> data
    Namespace(_dim=2, bar=[2, 3]<torch.float32>, foo=123)
    ```
    ----------------------------------------------------------------------
    """

    def __init__(self, dim: int):
        """
        # Parameters

        * `dim` - Length of first dimension for tensor values
        """
        self._dim = dim

    def __len__(self) -> int:
        """Returns number of items stored in this container, minus internal
        attributes like `_dim`.
        """
        return len(self.__dict__) - 1

    def __repr__(self) -> str:
        items = ""
        for k in sorted(self.__dict__.keys()):
            val = self.__dict__[k]
            if isinstance(val, Tensor):
                items += f", {k}={list(val.shape)}<{val.dtype}>"
            else:
                items += f", {k}={val}"
        return "Namespace(" + items.removeprefix(', ') + ")"

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Tensor):
            assert (self._dim == value.shape[0])
        super().__setattr__(name, value)

