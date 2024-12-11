from pydantic import BaseModel, ConfigDict


class Bucket(BaseModel):
    """Configuration for a bucket, including name and access credentials."""

    model_config = ConfigDict(str_min_length=1, str_strip_whitespace=True)

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
