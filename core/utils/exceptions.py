class GAGSError(Exception):
    """Base exception for all GAGS framework errors."""
    pass

class SimulationError(GAGSError):
    """Raised when simulation parameters are invalid or execution fails."""
    pass

class GovernanceViolation(GAGSError):
    """Raised when a model fails to meet the safety/equity thresholds."""
    pass