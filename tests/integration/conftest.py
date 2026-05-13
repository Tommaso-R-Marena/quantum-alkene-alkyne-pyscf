"""
Integration tests touch real PySCF / OpenFermion. The ``unit/`` tests mock
those packages via ``sys.modules.setdefault`` which pollutes the import
table for the rest of the test run. We force every integration test to run
in its own forked subprocess so that real packages can be imported cleanly.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.forked)
