# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import asyncio
import functools
import time
from collections.abc import Awaitable
from functools import partial
from typing import (
    Any,
    Callable,
    TypeVar,
)

import tplr
from tplr import exceptions

_R = TypeVar("_R")  # wrapped function return type
_E = TypeVar("_E")  # on_error_return return type
_OnErrReturn = TypeVar("_OnErrReturn")  # on_error_return callable's return type


def retry_on_failure(
    *,
    retries: int = -1,
    timeout: int = -1,
    delay: float = 0.5,
):
    """
    An asynchronous decorator factory to handle retry logic with a timeout, a
    fixed number of retries, and an optional grace period based on a maximum time.

    This decorator assumes the decorated function will return an object with a
    '.success' attribute (boolean) and a '.status' attribute (string).
    The decorator will retry if '.success' is False and '.status' is not 'TOO_LATE'.

    Args:
        retries: The maximum number of retries. -1 for infinite retries.
        timeout: The maximum time in seconds to keep retrying. -1 for no timeout.
        delay: The time in seconds to wait between retries.
    """

    def decorator(
        func: Callable[..., Awaitable[_R | None]],
    ) -> Callable[..., Awaitable[_R | None]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> _R | None:
            """
            The wrapper function that implements the retry logic.
            It calls the decorated function, and retries based on the result
            and decorator arguments.
            """
            start_time = time.time()
            attempts = 0
            func_name = func.__name__

            while True:
                base_message = f"Attempt {attempts + 1} on '{func_name}'"
                # Check for max retries
                if retries != -1 and attempts >= retries:
                    tplr.logger.info(
                        f"{base_message}: Max retries ({retries}) exceeded. Stopping."
                    )
                    return None

                # Check for global timeout
                if timeout != -1 and time.time() >= start_time + timeout:
                    tplr.logger.info(
                        f"{base_message}: Global timeout ({timeout}s) reached. Stopping."
                    )
                    return None

                # Call the decorated function (the core business logic)
                attempts += 1
                tplr.logger.debug(
                    f"Attempt {attempts} of {retries if retries != -1 else 'infinite'} on '{func_name}': Calling function..."
                )
                result = await func(*args, **kwargs)

                if timeout != -1 and time.time() >= start_time + timeout:
                    tplr.logger.info(
                        f"{base_message}: Completed but TOO_LATE. Stopping retries"
                    )
                    return None

                if result:  # and not result too_early/too_late?
                    tplr.logger.info(f"{base_message}: Function call successful.")
                    return result

                else:
                    tplr.logger.info(f"{base_message} failed. Retrying in {delay}s...")
                    await asyncio.sleep(delay)

        return wrapper

    return decorator


def async_exception_catcher(
    exception_handler: Callable[[_E, str], Any],
    on_error_return: Callable[[Exception], _OnErrReturn | None] = lambda e: None,
    on_error_raise: bool = False,
) -> Callable[
    [Callable[..., Awaitable[_R]]], Callable[..., Awaitable[_R | _OnErrReturn]]
]:
    """
    A decorator for asynchronous functions that wraps the function
    in a try...except block and uses the centralized exception handler.
    """

    def decorator(
        func: Callable[..., Awaitable[_R]],
    ) -> Callable[..., Awaitable[_R | _OnErrReturn]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> _R | _E:
            try:
                return await func(*args, **kwargs)
            except Exception as e:  # noqa: W0718
                handler_output = exception_handler(e, func.__name__)
                if handler_output:
                    return handler_output

                if on_error_raise:
                    raise

                return on_error_return(e)

        return wrapper

    return decorator


# def sync_exception_catcher(
#     exception_handler: Callable,
#     on_error_return: Callable[[Exception], _E] = lambda e: None,
#     on_error_raise: bool = False,
# ) -> Callable[[Callable[..., _R]], Callable[..., _R, | _E]]:
#     """
#     A decorator for synchronous functions that wraps the function
#     in a try...except block and uses the centralized exception handler.
#     """

#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args: Any, **kwargs: Any) -> _R | _E:
#             try:
#                 return func(*args, **kwargs)
#             except Exception as e:  # noqa: W0718
#                 purge = exception_handler(e, func.__name__)
#                 if purge:
#                     raise NotImplementedError(
#                         "Purge is async; not supported for sync wrapper"
#                     ) from e

#                 if on_error_raise:
#                     raise

#                 return on_error_return(e)
#         return wrapper
#     return decorator


async_s3_exception_catcher = partial(
    async_exception_catcher, exceptions.handle_s3_exceptions
)
# s3_exception_catcher = partial(sync_exception_catcher, exceptions.handle_s3_exceptions)
general_exception_catcher = partial(
    async_exception_catcher, exceptions.handle_general_exceptions
)
