from typing import List


class ValidationResult:
    def __init__(self, valid: bool, errors: List[str]):
        self.valid = valid
        self.errors = errors

    def __repr__(self):
        return f"ValidationResult(valid={self.valid}, errors={self.errors})"
