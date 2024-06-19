class ModerationPiiError(Exception):
    """Exception raised if PII entities are detected.

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self, message: str = "The prompt contains PII entities and cannot be processed"
    ):
        self.message = message
        super().__init__(self.message)


class ModerationToxicityError(Exception):
    """Exception raised if Toxic entities are detected.

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self, message: str = "The prompt contains toxic content and cannot be processed"
    ):
        self.message = message
        super().__init__(self.message)


class ModerationPromptSafetyError(Exception):
    """Exception raised if Unsafe prompts are detected.

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message: str = ("The prompt is unsafe and cannot be processed"),
    ):
        self.message = message
        super().__init__(self.message)
