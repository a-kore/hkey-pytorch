from torch import Tensor
import torch
from typing import List


def unpad_sequence(
    padded_sequences: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
) -> List[Tensor]:
    r"""Unpad padded Tensor into a list of variable length Tensors

    ``unpad_sequence`` unstacks padded Tensor into a list of variable length Tensors.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> sequences = [a, b, c]
        >>> padded_sequences = pad_sequence(sequences)
        >>> lengths = torch.as_tensor([v.size(0) for v in sequences])
        >>> unpadded_sequences = unpad_sequence(padded_sequences, lengths)
        >>> torch.allclose(sequences[0], unpadded_sequences[0])
        True
        >>> torch.allclose(sequences[1], unpadded_sequences[1])
        True
        >>> torch.allclose(sequences[2], unpadded_sequences[2])
        True

    Args:
        padded_sequences (Tensor): padded sequences.
        lengths (Tensor): length of original (unpadded) sequences.
        batch_first (bool, optional): whether batch dimension first or not. Default: False.

    Returns:
        a list of :class:`Tensor` objects
    """

    unpadded_sequences = []

    if not batch_first:
        padded_sequences.transpose_(0, 1)

    max_length = padded_sequences.shape[1]
    idx = torch.arange(max_length, device=padded_sequences.device)

    for seq, length in zip(padded_sequences, lengths):
        mask = idx < length
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences


def model_size(model):
    """Returns the size of the model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
