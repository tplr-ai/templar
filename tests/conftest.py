# Register the asyncio marker
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as requiring async")
