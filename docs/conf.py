# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

try:
    import chdb
except ImportError:
    chdb = None

autodoc_mock_imports = [
    "_chdb",
    "chdb._chdb",
    "pandas",
    "pyarrow",
    "chdb.dataframe",
]
autodoc_type_aliases = {
    'datetime': 'datetime.datetime',
}

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    "sphinx.ext.intersphinx",
]

source_suffix = '.rst'
master_doc = 'index'
project = 'chDB'
year = '2024'
author = 'chDB Team'
copyright = '{0}, {1}'.format(year, author)

if chdb and hasattr(chdb, '__version__'):
    version = release = chdb.__version__
else:
    version = release = '3.7.0'

github_repo_url = 'https://github.com/chdb-io/chdb'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': (f'{github_repo_url}/issues/%s', '#%s'),
    'pr': (f'{github_repo_url}/pull/%s', 'PR #%s'),
    'commit': (f"{github_repo_url}/commit/%s", "%s"),
}

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'alabaster'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
    '**': [
        'about.html',
        'searchbox.html',
        'globaltoc.html',
        'sourcelink.html'
    ],
}
html_theme_options = {
    'description': 'In-process OLAP SQL Engine powered by ClickHouse',
    'github_button': True,
    'github_user': 'chdb-io',
    'github_repo': 'chdb',
}

html_short_title = '%s-%s' % (project, version)

# Static files directory
html_static_path = ['_static']

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

autodoc_default_options = {
    'undoc-members': False,
    'show-inheritance': True,
}

# Add configuration to handle import issues during documentation build
autodoc_preserve_defaults = True
suppress_warnings = ['autodoc.mocked_object']

intersphinx_mapping = {
    # Temporarily disable problematic external documentation links
    # to avoid SSL/HTTP errors during documentation build
    # "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    # PyArrow official documentation (not ReadTheDocs)
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

# Configure intersphinx timeout and disable SSL verification if needed
intersphinx_timeout = 30
intersphinx_disabled_reftypes = []
