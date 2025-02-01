"""Base mock classes and utilities"""
import torch
from unittest.mock import MagicMock, AsyncMock
from types import SimpleNamespace

class BaseMock:
    """Base class for all mocks with common utilities"""
    @classmethod
    def create(cls, **kwargs):
        """Factory method to create mock instances with custom attributes"""
        instance = cls()
        for key, value in kwargs.items():
            setattr(instance, key, value)
        return instance

    def __getattr__(self, name):
        """Handle any unexpected attribute access"""
        return None 