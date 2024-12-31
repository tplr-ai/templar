# The MIT License (MIT)
# © 2024 templar.tech

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
# fmt: off

# Global imports
from pydantic import BaseModel


class Bucket(BaseModel):
    """Configuration for a bucket, including name and access credentials."""

    def __hash__(self):
        # Use all fields to generate a unique hash
        return hash(
            (self.name, self.account_id, self.access_key_id, self.secret_access_key)
        )

    def __eq__(self, other):
        # Compare all fields to determine equality
        if isinstance(other, Bucket):
            return self.dict() == other.dict()
        return False

    name: str
    account_id: str
    access_key_id: str
    secret_access_key: str
    class Config:
        str_min_length = 1
        str_strip_whitespace = True
