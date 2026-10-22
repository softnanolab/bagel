"""
Ways of reaching mosaic from bagel. All of them speak the request/response format of :mod:`.worker`.

* :class:`ModalMosaicBackend` - a deployed Modal class (``modal_service.py``). Needs a Modal account, no local GPU.
* :class:`SubprocessMosaicBackend` - a child process running ``worker.py`` in the mosaic environment, on the GPU of the
  machine bagel runs on: a local GPU, or a SLURM/PBS job (with :meth:`SubprocessMosaicBackend.apptainer` to run the
  published image).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time
from typing import Any, Protocol, Sequence

import numpy as np

from .adapter import MosaicPrediction
from .errors import MosaicBackendError
from .models import MosaicModelSpec
from .worker import RESPONSE_PREFIX, decode_array

logger = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).with_name('worker.py')


class MosaicBackend(Protocol):
    """Anything that can run mosaic forward passes."""

    def predict(
        self,
        spec: MosaicModelSpec,
        target_sequences: list[str],
        design_sequences: list[str],
        structure: bool = False,
    ) -> list[MosaicPrediction]:
        """One forward pass per designed-chain sequence, all with the same fixed ``target_sequences``.

        With ``structure=True`` the predictions also carry the all-atom coordinates (``atom37_coords``, ``atom37_mask``).
        """
        ...

    def versions(self) -> dict[str, str]: ...


def _request(
    spec: MosaicModelSpec, target_sequences: list[str], design_sequences: list[str], structure: bool = False
) -> dict[str, Any]:
    return {
        'structure': structure,
        'op': 'predict',
        'model': spec.to_dict(),
        'target_sequences': list(target_sequences),
        'design_sequences': list(design_sequences),
    }


def _decode(response: dict[str, Any], n_expected: int) -> list[MosaicPrediction]:
    if 'error' in response:
        raise MosaicBackendError(f'mosaic backend failed: {response["error"]}')
    plddt, pae, ptm = (decode_array(response[k]) for k in ('plddt', 'pae', 'ptm'))
    if not (len(plddt) == len(pae) == len(ptm) == n_expected):
        raise MosaicBackendError(f'mosaic backend returned {len(plddt)} predictions for {n_expected} sequences')
    if not (np.isfinite(plddt).all() and np.isfinite(pae).all() and np.isfinite(ptm).all()):
        raise MosaicBackendError('mosaic backend returned non-finite confidences')
    coords = decode_array(response['atom37_coords']) if 'atom37_coords' in response else None
    mask = decode_array(response['atom37_mask']) if 'atom37_mask' in response else None
    return [
        MosaicPrediction(
            plddt=plddt[i],
            pae=pae[i],
            ptm=float(ptm[i]),
            atom37_coords=None if coords is None else coords[i],
            atom37_mask=None if mask is None else mask[i],
        )
        for i in range(n_expected)
    ]


class ModalMosaicBackend:
    """
    Client of the mosaic service deployed on Modal (see ``modal_service.py``).

    Parameters
    ----------
    app_name : str
        Name of the deployed Modal app.
    class_name : str
        Name of the Modal class inside the app.
    """

    def __init__(self, app_name: str = 'bagel-mosaic-screening', class_name: str = 'MosaicService') -> None:
        self.app_name = app_name
        self.class_name = class_name
        self._service: Any = None

    def _get(self) -> Any:
        if self._service is None:
            try:
                import modal

                self._service = modal.Cls.from_name(self.app_name, self.class_name)()
            except Exception as exc:
                raise MosaicBackendError(
                    f"Cannot reach the Modal app '{self.app_name}' (class '{self.class_name}'): {exc}. "
                    'Deploy it with `modal deploy src/bagel/mosaic_screening/modal_service.py`.'
                ) from exc
        return self._service

    def predict(
        self,
        spec: MosaicModelSpec,
        target_sequences: list[str],
        design_sequences: list[str],
        structure: bool = False,
    ) -> list[MosaicPrediction]:
        try:
            response = self._get().predict.remote(
                json.dumps(_request(spec, target_sequences, design_sequences, structure))
            )
        except MosaicBackendError:
            raise
        except Exception as exc:
            raise MosaicBackendError(f'Modal call failed: {exc}') from exc
        return _decode(response, len(design_sequences))

    def versions(self) -> dict[str, str]:
        # ``modal.Cls.from_name`` is lazy: an undeployed app is only reported by the first remote call, so that has to be
        # wrapped too (the integration tests rely on ``MosaicBackendError`` to skip when the app is not deployed).
        try:
            return dict(self._get().versions.remote())
        except MosaicBackendError:
            raise
        except Exception as exc:
            raise MosaicBackendError(
                f"Modal call failed for app '{self.app_name}' (class '{self.class_name}'): {exc}. "
                'Deploy it with `modal deploy src/bagel/mosaic_screening/modal_service.py`.'
            ) from exc


class SubprocessMosaicBackend:
    """
    Runs ``worker.py`` as a child process and talks to it in JSON lines (stdin/stdout). The process is started on first
    use and kept alive, so the models are loaded and the losses compiled once.

    Parameters
    ----------
    command : sequence of str
        Command that starts the worker, e.g. ``['/path/to/mosaic/.venv/bin/python', '/path/to/worker.py']``. See the
        constructors :meth:`local` and :meth:`apptainer`.
    env : dict[str, str], optional
        Extra environment variables of the child process (e.g. ``HF_HOME``, ``HF_TOKEN``).
    timeout : float
        Seconds to wait for one answer. The first call includes loading the weights and compiling, which takes minutes.
    """

    def __init__(self, command: Sequence[str], env: dict[str, str] | None = None, timeout: float = 3600.0) -> None:
        self.command = list(command)
        self.env = env or {}
        self.timeout = timeout
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()

    @classmethod
    def local(
        cls, mosaic_python: str | Path, env: dict[str, str] | None = None, **kwargs: Any
    ) -> 'SubprocessMosaicBackend':
        """Use a Python interpreter of an environment where mosaic is installed (``<mosaic>/.venv/bin/python``)."""
        return cls([str(mosaic_python), str(WORKER_SCRIPT)], env=env, **kwargs)

    @classmethod
    def apptainer(
        cls,
        image: str | Path,
        binds: Sequence[str] = (),
        env: dict[str, str] | None = None,
        worker_path: str = '/opt/svc/worker.py',
        python_path: str = '/opt/mosaic/.venv/bin/python',
        nv: bool = True,
        **kwargs: Any,
    ) -> 'SubprocessMosaicBackend':
        """
        Run the published image with Apptainer (SLURM/PBS jobs). ``binds`` are ``host:container`` pairs, typically the
        cache directory with the model weights, which compute nodes without internet need to have pre-populated.
        """
        command = ['apptainer', 'exec']
        if nv:
            command.append('--nv')
        for bind in binds:
            command += ['--bind', bind]
        command += [str(image), python_path, worker_path]
        return cls(command, env=env, **kwargs)

    def _start(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            logger.info(f'Starting mosaic worker: {shlex.join(self.command)}')
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # worker logs go to the parent's stderr
                text=True,
                env={**os.environ, **self.env},
                bufsize=1,
            )
            self._lines = queue.Queue()
            threading.Thread(target=self._read, args=(self._proc, self._lines), daemon=True).start()
        return self._proc

    @staticmethod
    def _read(proc: 'subprocess.Popen[str]', lines: 'queue.Queue[str | None]') -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # EOF: the process has exited

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            proc = self._start()
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(json.dumps(request) + '\n')
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise MosaicBackendError(f'mosaic worker is not running (exit code {proc.poll()})') from exc
            deadline = time.time() + self.timeout
            while True:
                try:
                    line = self._lines.get(timeout=max(0.0, deadline - time.time()))
                except queue.Empty:
                    proc.kill()
                    raise MosaicBackendError(f'mosaic worker did not answer within {self.timeout:.0f} s') from None
                if line is None:
                    raise MosaicBackendError(f'mosaic worker exited (exit code {proc.wait()}) before answering')
                if line.startswith(RESPONSE_PREFIX):
                    response: dict[str, Any] = json.loads(line[len(RESPONSE_PREFIX) :])
                    return response

    def predict(
        self,
        spec: MosaicModelSpec,
        target_sequences: list[str],
        design_sequences: list[str],
        structure: bool = False,
    ) -> list[MosaicPrediction]:
        return _decode(self._call(_request(spec, target_sequences, design_sequences, structure)), len(design_sequences))

    def versions(self) -> dict[str, str]:
        response = self._call({'op': 'versions'})
        if 'error' in response:
            raise MosaicBackendError(response['error'])
        return dict(response['versions'])

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
