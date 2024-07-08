import ctypes


def is_libcublas_available() -> bool:
    try:
        ctypes.CDLL("libcublas.so")
        return True
    except OSError:
        return False
