class ValidationResult:
    def __init__(self, valid: bool, errors: list[str]):
        self.valid = valid
        self.errors = errors

    def __repr__(self):
        return f"ValidationResult(valid={self.valid}, errors={self.errors})"