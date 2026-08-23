from prometheus_client import Counter

REQUEST_COUNT = Counter(
    "rca_request_total",
    "Total RCA Requests"
)

SUCCESS_COUNT = Counter(
    "rca_success_total",
    "Successful RCA Runs"
)

FAILURE_COUNT = Counter(
    "rca_failure_total",
    "Failed RCA Runs"
)