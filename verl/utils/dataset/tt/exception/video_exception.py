class VideoException(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"VideoException: {self.args[0]}"

class VideoCorruptException(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"VideoCorruptException: {self.args[0]}"