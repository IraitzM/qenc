"""Batched torch statevector simulation of the three families, used by the regime map.

The models reproduce the PennyLane classifiers of ``qenc.models`` gate for gate,
but simulate a whole batch as one tensor on CPU or CUDA, and expose the tangent
kernel and the linearized model through ``torch.func``.
"""
