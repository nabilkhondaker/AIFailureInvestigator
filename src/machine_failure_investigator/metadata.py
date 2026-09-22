"""Project metadata constants."""

from machine_failure_investigator.version import __version__

PROJECT_NAME = "AI Machine Failure Investigator"
PACKAGE_NAME = "machine_failure_investigator"
AUTHOR = "Nabil Khondaker"
LICENSE = "MIT"
VERSION = __version__

FAULT_LABELS = [
    "HEALTHY",
    "BEARING_DEGRADATION",
    "BEARING_FAILURE",
    "IMBALANCE",
    "MISALIGNMENT",
    "LOOSENESS",
    "GEAR_WEAR",
    "LUBRICATION_FAILURE",
    "OVERHEATING",
    "SHAFT_DEFECT",
    "SENSOR_FAULT",
    "UNKNOWN_ANOMALY",
]

SEVERITY_LABELS = {
    (0.0, 0.2): "normal",
    (0.2, 0.4): "early",
    (0.4, 0.6): "moderate",
    (0.6, 0.8): "severe",
    (0.8, 1.01): "critical",
}
