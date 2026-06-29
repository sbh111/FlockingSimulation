"""
Author: Saad Bhatti
Desc:
Make the CUDA runtime libraries from the `nvidia-*-cu11` pip wheels discoverable
on Windows before CuPy loads them.

The `cupy-cuda11x` wheel does not bundle the CUDA libraries (nvrtc, cublas, ...)
and only knows how to find a system CUDA Toolkit install. When the libraries come
from the separate `nvidia-*-cu11` pip wheels instead, their DLLs live in
``site-packages/nvidia/<lib>/bin`` and Windows won't search those locations on its
own. Importing this module registers each of those directories with the loader so
CuPy can resolve e.g. ``nvrtc64_112_0.dll``.

Import this BEFORE importing cupy. It is a no-op on non-Windows platforms and when
the nvidia wheels are not installed.
"""

import os
import sys
import glob


def register_cuda_dlls():
    if sys.platform != "win32":
        return []
    try:
        import nvidia
    except ImportError:
        return []

    base = os.path.dirname(nvidia.__file__)
    added = []
    for bindir in glob.glob(os.path.join(base, "*", "bin")):
        if os.path.isdir(bindir):
            try:
                os.add_dll_directory(bindir)
            except (OSError, AttributeError):
                pass
            # Also prepend to PATH for any loader path that consults it.
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
            added.append(bindir)
    return added


_registered = register_cuda_dlls()
