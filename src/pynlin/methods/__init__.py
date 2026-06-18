"""Reusable NLIN method implementations.

This package is the public library layer for method-specific code. Analysis
scripts should use these modules for TD, MC, and PCFM computations, keeping
study orchestration, plotting, and caching outside ``pynlin``.
"""

__all__ = ["td", "mc", "pcfm"]