# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../pysigma'))

# Import test. 
import pysigma


# -- Project information -----------------------------------------------------

project = 'PySigma'
copyright = '2020, Jincheng Zhou, Yunzhe Wang, Volkan Ustun, Paul Rosenbloom @ USC Institute for Creative Technologies'
author = 'Jincheng Zhou, Yunzhe Wang, Volkan Ustun, Pawant to as or your custom'
# ones.
# sphinx.ext.autodoc - Automatically importing docstrings from source module
# sphinx_rtd_theme - The official Read-the-Docs theme
# sphinx.ext.napoleon - Preprocess Numpy/Google style docstrings
# sphinx.ext.viewcode - Insert links to generated source code HTML 
extensions = [
	'sphinx.ext.autodoc',
	'sphinx_rtd_theme',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode', 
	'sphinx.ext.autosummary', 
	'sphinx.ext.todo'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Explicitly fix the master doc to index.rst
master_doc = 'index'

# For autodoc: Output entities in the order specified in the source code
autodoc_member_order = 'bysource'

# Allow showing TODO items in the docs
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation" : False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_sidebars = {
# 	'**': ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']
# }