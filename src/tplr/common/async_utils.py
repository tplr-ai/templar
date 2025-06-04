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
from typing import Any, Callable, TypeVar

import tplr

T = TypeVar('T')


async def retry_call(
    func: Callable[..., T], 
    *args, 
    attempts: int = 3, 
    delay: int = 1, 
    context: str = "", 
    **kwargs
) -> T | None:
    """
    Calls an async or sync function with retries.

    Args:
        func: An async or sync function to call
        *args: Positional arguments to pass to func
        attempts: Number of retry attempts
        delay: Delay between attempts in seconds  
        context: Context description for logging
        **kwargs: Keyword arguments to pass to func

    Returns:
        The result of func(*args, **kwargs) or None if all attempts fail
    """
    for attempt in range(attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                def wrapper(*w_args, **w_kwargs):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(func(*w_args, **w_kwargs))
                    finally:
                        loop.close()

                return await asyncio.to_thread(wrapper, *args, **kwargs)
            else:
                return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as e:
            tplr.logger.error(
                f"Attempt {attempt + 1}/{attempts} failed for {context}: {e}"
            )
            if attempt < attempts - 1:  # Don't sleep on the last attempt
                await asyncio.sleep(delay)
    
    tplr.logger.error(f"Failed to complete {context} after {attempts} attempts.")
    return None 