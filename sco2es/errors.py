class ModelAssumptionError(Exception):
    """Exception raised when a model assumption is not met."""


class StopCriterionError(Exception):
    """Exception raised when the charge/discharge stopping criterion is not met within the allowable time."""
