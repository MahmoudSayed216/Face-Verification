class Logger:
    # Activation flags
    debug_active = False
    log_active = False
    checkpoint_active = False

    # Counters
    debug_count = 0
    log_count = 0
    checkpoint_count = 0

    # ANSI color codes
    COLORS = {
        'red': "\033[31m",
        'green': "\033[32m",
        'cyan': "\033[36m",
        'reset': "\033[0m"
    }

    @classmethod
    def _log(cls, level: str, color: str, message: str, obj=None, endl=False):
        """Internal generic logging method."""
        # Pick counter based on level
        if level == "DEBUGGER":
            counter = cls.debug_count
            if not cls.debug_active:
                return
        elif level == "LOGGER":
            counter = cls.log_count
            if not cls.log_active:
                return
        elif level == "CHECKPOINT":
            counter = cls.checkpoint_count
            if not cls.checkpoint_active:
                return
        else:
            raise ValueError(f"Unknown log level: {level}")

        # Build message
        counter_str = str(counter).rjust(4, '0')
        prefix = f"{cls.COLORS[color]}[{level}:{counter_str}]{cls.COLORS['reset']}"
        if obj is not None:
            message = f" {message}: {obj}"

        # Print
        print(f"{prefix}{message}", end="\n\n" if endl else "\n")

        # Update counter
        if level == "DEBUGGER":
            cls.debug_count += 1
        elif level == "LOGGER":
            cls.log_count += 1
        elif level == "CHECKPOINT":
            cls.checkpoint_count += 1

    # Public APIs
    @classmethod
    def debug(cls, message: str, obj=None, endl=False):
        cls._log("DEBUGGER", "red", message, obj, endl)

    @classmethod
    def log(cls, message: str, obj=None, endl=False):
        cls._log("LOGGER", "green", message, obj, endl)

    @classmethod
    def checkpoint(cls, message: str, endl=False):
        cls._log("CHECKPOINT", "cyan", message, None, endl)
