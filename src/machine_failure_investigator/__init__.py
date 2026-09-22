"""
AI Machine Failure Investigator
================================

Research framework for investigating faults in rotating mechanical machinery
using multi-sensor observations, signal processing, and machine learning.

Author: Nabil Khondaker
"""

from machine_failure_investigator.version import __version__
from machine_failure_investigator.diagnosis.investigator import Investigator

__all__ = ["__version__", "Investigator"]
