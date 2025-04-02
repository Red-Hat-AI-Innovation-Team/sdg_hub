"""version information"""
try:
    from importlib.metadata import version

    __version__ = version("sdg-hub")
except ImportError:
    __version__ = "unknown"