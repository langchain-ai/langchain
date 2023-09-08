"""A simple progress bar for the console."""
from threading import Lock


class ProgressBar:
    """A simple progress bar for the console."""

    def __init__(self, total: int, length: int = 50):
        self.total = total
        self.length = length
        self.counter = 0
        self.lock = Lock()

    def increment(self) -> None:
        """Increment the counter and update the progress bar."""
        with self.lock:
            self.counter += 1
            self._print_bar()

    def _print_bar(self) -> None:
        """Print the progress bar to the console."""
        progress = self.counter / self.total
        arrow = "-" * int(round(progress * self.length) - 1) + ">"
        spaces = " " * (self.length - len(arrow))
        print(f"\r[{arrow + spaces}] {self.counter}/{self.total}", end="")
