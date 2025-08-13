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

# Global imports
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Bucket(BaseModel):
    """Configuration for a bucket, including name and access credentials."""

    def __hash__(self):
        # Use all fields to generate a unique hash
        return hash(
            (self.name, self.account_id, self.access_key_id, self.secret_access_key)
        )

    name: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)
    access_key_id: str = Field(..., min_length=1)
    secret_access_key: str = Field(..., min_length=1)

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )


class CommsGetResult(BaseModel):
    """A standard return type for the `get` function."""

    data: Optional[Dict[str, Any]] = Field(
        None, description="The data retrieved by the get function."
    )
    global_step: Optional[int] = Field(
        None, description="The global step associated with the data."
    )
    status: Literal["OK", "TOO_EARLY", "TOO_LATE", "NOT_FOUND", "ERROR"] = Field(
        "OK", description="The status of the get operation."
    )

    @property
    def success(self) -> bool:
        """Returns True if the operation was successful and returned data."""
        return self.status == "OK" and self.data is not None
