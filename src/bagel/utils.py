import pathlib
import re
from typing import Optional


def get_version_from_pyproject(pyproject_path: Optional[str] = None) -> Optional[str]:
    """Parse the version string from pyproject.toml."""
    if pyproject_path is None:
        path = pathlib.Path(__file__).parent.parent.parent / 'pyproject.toml'
    else:
        path = pathlib.Path(pyproject_path)
    version = None
    if path.exists():
        with open(path) as f:
            for line in f:
                m = re.match(r'version\s*=\s*[\'"]([^\'"]+)[\'"]', line)
                if m:
                    version = m.group(1)
                    break
    return version
