"""
Modal service that runs mosaic forward passes for :class:`~bagel.mosaic_screening.ModalMosaicBackend`.

It is built from ``docker/mosaic-screening/Dockerfile`` (the same recipe CI publishes to ghcr.io), so Modal, local GPU and
cluster runs use the same environment.

Deploy once, from the bagel environment, with the Modal profile you want to use active::

    modal deploy src/bagel/mosaic_screening/modal_service.py

Environment variables read at deploy time: ``MOSAIC_GPU`` (default ``A100-80GB``), ``MOSAIC_MAX_CONTAINERS`` (default 4),
``MOSAIC_CACHE_VOLUME`` (Modal volume for the model weights), ``MOSAIC_APP_NAME`` (default ``bagel-mosaic-screening``),
``MOSAIC_SCALEDOWN`` (seconds an idle container stays up, default 600). The Modal secret ``huggingface-secret`` must
provide a Hugging Face token (the ESMFold2 weights are gated).
"""

import json
import os
import pathlib
import shutil
import tempfile

import modal

APP_NAME = os.environ.get('MOSAIC_APP_NAME', 'bagel-mosaic-screening')
GPU = os.environ.get('MOSAIC_GPU', 'A100-80GB')
MAX_CONTAINERS = int(os.environ.get('MOSAIC_MAX_CONTAINERS', '4'))
SCALEDOWN = int(os.environ.get('MOSAIC_SCALEDOWN', '600'))
CACHE_VOLUME = os.environ.get('MOSAIC_CACHE_VOLUME', 'bagel-mosaic-screening-cache')

HERE = pathlib.Path(__file__).resolve().parent


def _build_context() -> str:
    """Minimal Docker build context (the Dockerfile COPYs two files), so Modal does not upload the repository."""
    REPO = HERE.parents[2]
    context = pathlib.Path(tempfile.mkdtemp(prefix='bagel_mosaic_ctx_'))
    (context / 'docker/mosaic-screening').mkdir(parents=True)
    (context / 'src/bagel/mosaic_screening').mkdir(parents=True)
    shutil.copy(REPO / 'docker/mosaic-screening/Dockerfile', context / 'docker/mosaic-screening/Dockerfile')
    shutil.copy(REPO / 'docker/mosaic-screening/uv.lock', context / 'docker/mosaic-screening/uv.lock')
    shutil.copy(HERE / 'worker.py', context / 'src/bagel/mosaic_screening/worker.py')
    return str(context)


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
if modal.is_local():
    _context = _build_context()
    image = (
        modal.Image.from_dockerfile(
            str(pathlib.Path(_context) / 'docker/mosaic-screening/Dockerfile'), context_dir=_context
        )
        # Modal's runtime must be importable by the interpreter that runs the worker: the mosaic environment's.
        .run_commands('uv pip install --python /opt/mosaic/.venv/bin/python modal')
        .env({'PATH': '/opt/mosaic/.venv/bin:/usr/local/bin:/usr/bin:/bin', 'PYTHONPATH': '/opt/svc'})
    )
else:  # inside the container the image is already built; the repository is not there
    image = modal.Image.debian_slim()


@app.cls(
    image=image,
    gpu=GPU,
    volumes={'/cache': cache_volume},
    secrets=[modal.Secret.from_name('huggingface-secret')],
    timeout=3600,
    scaledown_window=SCALEDOWN,
    max_containers=MAX_CONTAINERS,
)
class MosaicService:
    @modal.enter()
    def load(self) -> None:
        from worker import MosaicWorker

        self.worker = MosaicWorker()

    @modal.method()
    def predict(self, request_json: str) -> dict:
        """One batched forward pass per call; ``request_json`` is the ``predict`` request of ``worker.py``."""
        response = self.worker.handle(json.loads(request_json))
        cache_volume.commit()  # persist newly downloaded weights
        return response

    @modal.method()
    def versions(self) -> dict:
        return self.worker.versions()
