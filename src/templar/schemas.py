from pydantic import BaseModel


class Bucket(BaseModel):
    """Configuration for a bucket, including name and access credentials."""

    def __hash__(self):
        return hash(self.account_id)

    name: str
    account_id: str
    access_key_id: str
    secret_access_key: str

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
