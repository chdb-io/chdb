# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

try:
    import chdb
except ImportError:
    chdb = None

autodoc_mock_imports = ["_chdb"]
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
    version = release = '3.6.0'

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

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

autodoc_default_options = {
    'undoc-members': False
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}
