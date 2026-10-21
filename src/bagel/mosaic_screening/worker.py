"""
The part of the mosaic screening that runs *inside the mosaic environment* (JAX, torch, mosaic).

Mosaic (torch 2.11, JAX) and bagel's pinned boileroom (torch < 2.7) cannot be installed into one Python environment, so
mosaic always runs as a separate service: a Modal class (``modal_service.py``), or a subprocess that speaks JSON lines
over stdin/stdout (``python worker.py``; used on a local GPU and inside an Apptainer image on SLURM/PBS clusters).

This file is deliberately standalone: it imports only the standard library and numpy at module level (jax and mosaic are
imported lazily), so bagel can import it for the wire format while the mosaic environment can run it as a script
without bagel installed.

Wire format
-----------
Request  ``{"op": "predict", "model": <MosaicModelSpec dict>, "target_sequences": [...], "design_sequences": [...],
         "structure": bool}`` (``structure`` also returns the atom37 coordinates and mask)
         ``{"op": "versions"}`` / ``{"op": "ping"}``
Response ``{"plddt": A, "pae": A, "ptm": A, "seconds": float, ["atom37_coords": A, "atom37_mask": A]}`` where each ``A`` is ``{"shape": [...], "b64": str}``
         (little-endian float32), with a leading batch axis, in mosaic token order (designed chain first).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Any

import numpy as np

RESPONSE_PREFIX = 'BAGEL_MOSAIC_RESPONSE '
MAX_BATCH = 16  # sequences per forward; larger requests are split (GPU memory)


def encode_array(array: Any) -> dict[str, Any]:
    a = np.ascontiguousarray(np.asarray(array), dtype='<f4')
    return {'shape': list(a.shape), 'b64': base64.b64encode(a.tobytes()).decode('ascii')}


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(payload['b64'])
    return np.frombuffer(raw, dtype='<f4').reshape(payload['shape']).astype(np.float64)


class MosaicWorker:
    """Holds the loaded mosaic models and the per-problem feature packs; ``predict`` is the only expensive call."""

    def __init__(self) -> None:
        for name in ('HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN', 'HF_API_TOKEN'):
            if name in os.environ and 'HF_TOKEN' not in os.environ:
                os.environ['HF_TOKEN'] = os.environ[name]
        self._models: dict[Any, Any] = {}
        self._features: dict[Any, Any] = {}
        self._batch_fn: Any = None

    # -- models and features -------------------------------------------------------------------------------------
    def _model(self, spec: dict[str, Any]) -> Any:
        key = (spec['model'], spec['checkpoint'] if spec['model'] == 'esmfold2' else '')
        if key not in self._models:
            if spec['model'] == 'esmfold2':
                from mosaic.models.esmfold2 import ESMFold2Fast, ESMFold2Full

                self._models[key] = {'full': ESMFold2Full, 'fast': ESMFold2Fast}[spec['checkpoint']]()
            else:
                from mosaic.models.boltz2 import Boltz2

                self._models[key] = Boltz2()
        return self._models[key]

    def _features_for(self, spec: dict[str, Any], design_length: int, targets: list[str]) -> Any:
        key = (spec['model'], spec['checkpoint'], design_length, tuple(targets))
        if key not in self._features:
            from mosaic.structure_prediction import TargetChain

            model = self._model(spec)
            chains = [TargetChain(s, use_msa=False) for s in targets]  # empty MSA: no MSA server, deterministic
            features, _ = model.binder_features(design_length, chains)
            self._features[key] = features
        return self._features[key]

    def _batch(self) -> Any:
        if self._batch_fn is None:
            import equinox as eqx
            import jax
            from mosaic.losses.structure_prediction import predicted_tm_score

            # Defined once: a jitted function created inside the request would be recompiled every time.
            @eqx.filter_jit
            def run(model: Any, features: Any, xs: Any, key: Any, recycling_steps: int, sampling_steps: int) -> Any:
                def one(x: Any) -> Any:
                    out = model.model_output(
                        PSSM=x,
                        features=features,
                        recycling_steps=recycling_steps,
                        sampling_steps=sampling_steps,
                        key=key,
                    )
                    ptm = predicted_tm_score(logits=out.pae_logits, bin_centers=out.pae_bins).max()
                    return out.plddt, out.pae, ptm, out.atom37_coords, out.atom37_mask

                return jax.vmap(one)(xs)

            self._batch_fn = run
        return self._batch_fn

    # -- requests ------------------------------------------------------------------------------------------------
    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        import jax
        import jax.numpy as jnp
        from mosaic.common import tokenize

        t0 = time.time()
        spec = request['model']
        sequences = list(request['design_sequences'])
        targets = list(request['target_sequences'])
        design_length = len(sequences[0])
        if any(len(s) != design_length for s in sequences):
            raise ValueError('All design sequences of a request must have the same length.')
        model = self._model(spec)
        features = self._features_for(spec, design_length, targets)
        run = self._batch()
        key = jax.random.key(int(request.get('seed', 0)))
        plddt, pae, ptm, coords, mask = [], [], [], [], []
        for start in range(0, len(sequences), MAX_BATCH):
            chunk = sequences[start : start + MAX_BATCH]
            xs = jnp.stack([jax.nn.one_hot(tokenize(s), 20) for s in chunk])
            p, a, t, c, m = run(model, features, xs, key, int(spec['recycling_steps']), int(spec['sampling_steps']))
            plddt.append(np.asarray(p))
            pae.append(np.asarray(a))
            ptm.append(np.asarray(t))
            if request.get('structure'):
                coords.append(np.asarray(c))
                mask.append(np.asarray(m))
        response = {
            'plddt': encode_array(np.concatenate(plddt)),
            'pae': encode_array(np.concatenate(pae)),
            'ptm': encode_array(np.concatenate(ptm)),
            'seconds': time.time() - t0,
        }
        if coords:
            response['atom37_coords'] = encode_array(np.concatenate(coords))
            response['atom37_mask'] = encode_array(np.concatenate(mask))
        return response

    def versions(self) -> dict[str, str]:
        import importlib.metadata as md

        out = {}
        for name in ('mosaic', 'jax', 'jaxlib', 'equinox', 'torch', 'esmjfold2', 'joltz'):
            try:
                out[name] = md.version(name)
            except md.PackageNotFoundError:
                out[name] = 'not installed'
        out['mosaic_commit'] = os.environ.get('MOSAIC_COMMIT', 'unknown')
        return out

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get('op', 'predict')
        if op == 'ping':
            return {'ok': True}
        if op == 'versions':
            return {'versions': self.versions()}
        if op == 'predict':
            return self.predict(request)
        raise ValueError(f'Unknown op {op!r}')


class FakeWorker:
    """
    Deterministic stand-in with the same interface, used by the tests of the subprocess transport (and to check the
    wiring on machines without a GPU). pLDDT and PAE are simple functions of the designed sequence.
    """

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get('op', 'predict')
        if op in ('ping', 'versions'):
            return {'ok': True} if op == 'ping' else {'versions': {'mosaic_commit': 'fake'}}
        targets = request['target_sequences']
        n_target = sum(len(s) for s in targets)
        plddt, pae, ptm = [], [], []
        for seq in request['design_sequences']:
            n = len(seq) + n_target
            hydrophobic = np.array([c in 'AILMFVWY' for c in seq] + [False] * n_target, dtype=float)
            plddt.append(0.5 + 0.4 * hydrophobic.mean() * np.ones(n))
            pae.append(np.full((n, n), 20.0 - 10.0 * hydrophobic.mean()))
            ptm.append(0.4 + 0.4 * hydrophobic.mean())
        response = {
            'plddt': encode_array(np.array(plddt)),
            'pae': encode_array(np.array(pae)),
            'ptm': encode_array(np.array(ptm)),
            'seconds': 0.0,
        }
        if request.get('structure'):  # a helix-like trace with four backbone atoms and CB per residue
            n = len(request['design_sequences'][0]) + n_target
            t = np.arange(n)
            ca = np.stack([2.3 * np.cos(1.745 * t), 2.3 * np.sin(1.745 * t), 1.5 * t], axis=-1)
            coords = np.zeros((n, 37, 3))
            mask = np.zeros((n, 37))
            for slot, offset in (
                (0, (-1.0, 0.0, -0.5)),
                (1, (0.0, 0.0, 0.0)),
                (2, (1.0, 0.0, 0.5)),
                (4, (1.5, 0.5, 0.8)),
                (3, (0.0, 1.5, 0.0)),
            ):
                coords[:, slot] = ca + np.array(offset)
                mask[:, slot] = 1.0
            batch = len(request['design_sequences'])
            response['atom37_coords'] = encode_array(np.broadcast_to(coords, (batch, n, 37, 3)))
            response['atom37_mask'] = encode_array(np.broadcast_to(mask, (batch, n, 37)))
        return response


def serve_stdio(worker: Any) -> None:
    """Answer JSON-line requests on stdin with ``RESPONSE_PREFIX``-tagged JSON lines on the original stdout."""
    out = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
    sys.stdout = sys.stderr  # anything the libraries print must not corrupt the channel
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = worker.handle(json.loads(line))
        except Exception as exc:  # reported to the client, which raises MosaicBackendError
            import traceback

            traceback.print_exc()
            response = {'error': f'{type(exc).__name__}: {exc}'}
        out.write(RESPONSE_PREFIX + json.dumps(response) + '\n')
        out.flush()


if __name__ == '__main__':
    serve_stdio(FakeWorker() if '--fake' in sys.argv else MosaicWorker())
