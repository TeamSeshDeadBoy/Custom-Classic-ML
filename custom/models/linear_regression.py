from custom.metrics import regression_metrics

class summ:
    def __init__(self) -> None:
        self.value = regression_metrics.MSE()
        pass