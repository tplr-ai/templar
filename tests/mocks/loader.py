"""
Mock loader for testing evaluation.
This loader simply wraps a list of pre-defined batches.
"""
class MockLoader:
    def __init__(self, batches):
        self.batches = batches
      
    def __iter__(self):
        return iter(self.batches)
