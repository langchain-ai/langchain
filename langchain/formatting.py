from string import Formatter


class StrictFormatter(Formatter):
    """A subclass of formatter that checks for extra keys."""

    def check_unused_args(self, used_args, args, kwargs):
        extra = set(kwargs).difference(used_args)
        if extra:
            raise KeyError(extra)

    def vformat(self, format_string, args, kwargs):
        if len(args) > 0:
            raise ValueError(
                "No arguments should be provided, "
                "everything should be passed as keyword arguments."
            )
        return super().vformat(format_string, args, kwargs)


formatter = StrictFormatter()
