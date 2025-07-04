import pathlib
import re

def get_version_from_pyproject(pyproject_path=None):
    """Parse the version string from pyproject.toml."""
    if pyproject_path is None:
        pyproject_path = pathlib.Path(__file__).parent.parent.parent / 'pyproject.toml'
    else:
        pyproject_path = pathlib.Path(pyproject_path)
    version = None
    if pyproject_path.exists():
        with open(pyproject_path) as f:
            for line in f:
                m = re.match(r'version\s*=\s*[\'"]([^\'"]+)[\'"]', line)
                if m:
                    version = m.group(1)
                    break
    return version
