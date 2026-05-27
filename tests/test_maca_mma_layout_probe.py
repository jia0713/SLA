import os
import subprocess
from pathlib import Path

import pytest


def _maca_path():
    maca_path = os.environ.get("MACA_PATH")
    if maca_path:
        return Path(maca_path)
    default = Path("/opt/maca")
    if default.exists():
        return default
    return None


@pytest.mark.skipif(_maca_path() is None, reason="MACA_PATH is not available")
def test_maca_mma_16x16x16_layout_probe(tmp_path):
    maca_path = _maca_path()
    assert maca_path is not None

    mxcc = maca_path / "mxgpu_llvm" / "bin" / "mxcc"
    if not mxcc.exists():
        pytest.skip(f"mxcc not found at {mxcc}")

    source = Path(__file__).with_name("maca_mma_layout_probe.cpp")
    binary = tmp_path / "maca_mma_layout_probe"

    compile_cmd = [
        str(mxcc),
        "-x",
        "maca",
        "-offload-arch",
        "native",
        str(source),
        "-o",
        str(binary),
        f"--maca-path={maca_path}",
    ]
    subprocess.run(compile_cmd, check=True, text=True, capture_output=True)

    env = os.environ.copy()
    lib_path = str(maca_path / "lib")
    env["LD_LIBRARY_PATH"] = (
        lib_path
        if not env.get("LD_LIBRARY_PATH")
        else lib_path + os.pathsep + env["LD_LIBRARY_PATH"]
    )
    result = subprocess.run([str(binary)], check=True, text=True, capture_output=True, env=env)

    assert "MACA MMA layout probe PASS" in result.stdout
