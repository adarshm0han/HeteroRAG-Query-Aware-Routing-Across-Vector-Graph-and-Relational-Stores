"""Secret-free failure classification for stopping a batch on infrastructure loss."""

import urllib.error


def failure_details(error):
    reason = getattr(error, "reason", None)
    result = {"exception_type": type(error).__name__}
    for label, value in (
        ("errno", getattr(error, "errno", None)),
        ("winerror", getattr(error, "winerror", None)),
        ("reason_errno", getattr(reason, "errno", None)),
    ):
        if type(value) is int:
            result[label] = value
    if reason is not None:
        result["reason_type"] = type(reason).__name__
    status = (
        getattr(error, "code", None)
        if isinstance(error, urllib.error.HTTPError)
        else getattr(error, "status", None)
    )
    if type(status) is int:
        result["http_status"] = status
    names = {type(error).__name__, type(reason).__name__}
    infrastructure = (
        isinstance(
            error,
            (urllib.error.URLError, TimeoutError, ConnectionError, PermissionError),
        )
        or bool(
            names
            & {
                "MaxRetryError",
                "ServiceUnavailable",
                "SessionExpired",
                "OperationalError",
                "ConnectTimeout",
                "ReadTimeout",
                "ConnectionTimeout",
                "ProtocolError",
                "RemoteDisconnected",
                "IncompleteRead",
                "ConnectionResetError",
                "NewConnectionError",
                "ReadTimeoutError",
                "ConnectTimeoutError",
                "SSLError",
            }
        )
        or (type(status) is int and (status >= 500 or status in (401, 403, 408, 429)))
    )
    return {**result, "infrastructure": infrastructure}
