# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pynlin'
copyright = '2025, Gianluca Marcon, Francesco Lorenzi'
author = 'Gianluca Marcon, Francesco Lorenzi'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

autodoc_mock_imports = [
    "numba",
    "torch",
    "cvxpy",
    "plotly",
    "watchdog",
    "loguru",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# Prefer local MathJax if available (useful for offline builds), otherwise fall back to CDN.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
_mathjax_local = os.path.join(os.path.dirname(__file__), "_static", "mathjax", "es5", "tex-mml-chtml.js")
if os.path.exists(_mathjax_local):
    mathjax_path = "mathjax/es5/tex-mml-chtml.js"

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
