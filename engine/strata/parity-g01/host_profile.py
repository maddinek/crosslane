"""Optional site guard configuration. The reference site's profile stays private."""
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def profile():
    # An explicit missing profile is an error; a standalone public tree has none.
    path = Path(os.environ.get('CROSSLANE_HOST_PROFILE', ROOT/'config/crosslane-host.json'))
    private_tree = (ROOT/'engine/strata/parity-g01/window.py').exists()
    value = json.loads(path.read_text()) if path.exists() or 'CROSSLANE_HOST_PROFILE' in os.environ or private_tree else {}
    if not isinstance(value, dict):
        raise ValueError('host profile must be an object')
    return value


def expand_path(value):
    # Match the windows' original Path.home() semantics, including CPU fixtures.
    return Path.home()/str(value)[2:] if str(value).startswith('~/') else Path(value).expanduser()


def setting(key, default=''):
    value = os.environ.get('CROSSLANE_'+key.upper(), profile().get(key, default))
    return str(expand_path(value)) if value else ''


def lock_path():
    return Path(setting('lock', '~/.cache/crosslane/gpu.lock'))


def require_pause():
    p = profile()
    if p.get('production_checks'):
        if not p.get('pause_file') or not expand_path(p['pause_file']).is_file():
            raise ValueError('configured production pause file is absent')


def idle_inputs():
    return [expand_path(p) for p in profile().get('idle_inputs', [])]


def launch(command):
    p = profile()
    guard = setting('b65_guard')
    if p.get('require_b65_guard') and not guard:
        raise ValueError('host profile requires a B65 guard')
    lock = lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Precheck before a wrapper can act; fd8 is acquired only in its child.
    with lock.open('a') as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['CROSSLANE_LOCK'] = str(lock)
    child = ['bash', '-c', 'exec 8>"$CROSSLANE_LOCK"; flock --nonblock 8 || exit; exec "$@"', 'crosslane', *command]
    argv = [guard, *child] if guard else child
    os.execvp(argv[0], argv)


if __name__ == '__main__':
    launch(sys.argv[1:])
