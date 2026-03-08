"""
CardioSense — Custom exceptions and error-handling utilities.
"""

import logging
import functools
from flask import flash, redirect, url_for

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Base exception for service-layer errors."""

    def __init__(self, message="An unexpected error occurred.", status_code=500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class NotFoundError(ServiceError):
    """Raised when a requested resource does not exist."""

    def __init__(self, message="Resource not found."):
        super().__init__(message, status_code=404)


class ValidationError(ServiceError):
    """Raised when input data fails validation."""

    def __init__(self, message="Invalid input."):
        super().__init__(message, status_code=400)


def handle_route_errors(f):
    """Decorator for route functions — catches ServiceError and flashes a message."""

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ServiceError as e:
            logger.warning("ServiceError in %s: %s", f.__name__, e.message)
            flash(e.message, "danger")
            return redirect(url_for("landing"))
        except Exception:
            logger.exception("Unexpected error in %s", f.__name__)
            flash("An unexpected error occurred. Please try again.", "danger")
            return redirect(url_for("landing"))

    return decorated
