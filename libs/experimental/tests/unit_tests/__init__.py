import ctypes

def is_libcublas_available():
    try:
        ctypes.CDLL("libcublas.so")
        return True
    except OSError:
        return False