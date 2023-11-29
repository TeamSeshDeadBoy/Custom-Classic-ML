from custom.metrics import squared_errors

class summ:
    def __init__(self) -> None:
        self.value = squared_errors.MSE()
        pass