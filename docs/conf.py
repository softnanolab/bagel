# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from typing import List

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'bagel'
copyright = '2025, Ayham A, Jakub L, Stefano A'
author = 'Ayham A, Jakub L, Stefano A'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # Generates docs from docstrings
    'sphinx.ext.napoleon',    # Parses Google/NumPy style docstrings
    'sphinx.ext.viewcode',    # Adds links to source code
    'sphinx.ext.mathjax',     # For mathematical notation
    'sphinx.ext.intersphinx'  # Links to other docs (numpy, python, etc.)
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
