# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import logging
from rich.logging import RichHandler
from rich.highlighter import NullHighlighter

# Configure loguru logger
FORMAT = "%(message)s"
logging.basicConfig( 
    level=logging.INFO, 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[
        RichHandler(
            markup=True, 
            rich_tracebacks=True, 
            highlighter=NullHighlighter(),
            show_level=False,
            show_time=False,
            show_path=False
        )
    ]
)
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)
def debug():
    logger.setLevel(logging.DEBUG)
def trace():
    logger.setLevel(logging.TRACE)
# Log helper.
def T(): return time.time()
def P( w, d ): return f"[steel_blue]{w}[/steel_blue] ([grey63]{d:.2f}s[/grey63])"