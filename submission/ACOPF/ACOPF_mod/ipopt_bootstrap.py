
import os
import shutil
import subprocess
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_DIRS = [
    os.path.join(_HERE, "ipopt_testing"),
    os.path.join(os.path.dirname(os.path.dirname(_HERE)), "ipopt_testing"),
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
        "ipopt_testing",
    ),
]
_PREPARED = {}


def ipopt_dir():
    for directory in _CANDIDATE_DIRS:
        if os.path.exists(os.path.join(directory, "bin", "ipopt")):
            return directory
    return _CANDIDATE_DIRS[0]


def ipopt_lib():
    return os.path.join(ipopt_dir(), "lib")


def _runs(binary):
    try:
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = (
            ipopt_lib() + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
        return subprocess.run(
            [binary, "--version"],
            capture_output=True,
            env=environment,
            timeout=60,
        ).returncode == 0
    except Exception:
        return False


def _staging_dirs():
    out = []
    for directory in (
        os.environ.get("TMPDIR"),
        tempfile.gettempdir(),
        "/tmp",
        os.path.expanduser("~"),
        os.getcwd(),
    ):
        if directory and directory not in out:
            out.append(directory)
    return out


def prepare_ipopt():
    if "bin" in _PREPARED:
        return _PREPARED["bin"]

    original = os.path.join(ipopt_dir(), "bin", "ipopt")
    binary = original
    if os.path.exists(binary) and not os.access(binary, os.X_OK):
        try:
            os.chmod(binary, os.stat(binary).st_mode | 0o111)
        except OSError:
            pass

    if not (os.path.exists(binary) and os.access(binary, os.X_OK) and _runs(binary)):
        staged = None
        if os.path.exists(original):
            for base in _staging_dirs():
                try:
                    directory = os.path.join(base, "acopf_ipopt")
                    os.makedirs(directory, exist_ok=True)
                    destination = os.path.join(directory, "ipopt")
                    if not os.path.exists(destination) or os.path.getsize(
                        destination
                    ) != os.path.getsize(original):
                        shutil.copyfile(original, destination)
                    os.chmod(destination, 0o755)
                    if _runs(destination):
                        staged = destination
                        break
                except OSError:
                    continue
        binary = staged or shutil.which("ipopt") or original

    _PREPARED["bin"] = binary
    return binary


def bootstrap():
    os.environ["LD_LIBRARY_PATH"] = (
        ipopt_lib() + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")
    binary = prepare_ipopt()
    directory = os.path.dirname(binary)
    if directory and directory not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
    return binary, _runs(binary)
