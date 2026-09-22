"""Sensor models for synthetic measurement generation."""

from machine_failure_investigator.sensors.vibration import VibrationSensor
from machine_failure_investigator.sensors.temperature import TemperatureSensor
from machine_failure_investigator.sensors.current import CurrentSensor
from machine_failure_investigator.sensors.torque import TorqueSensor
from machine_failure_investigator.sensors.tachometer import TachometerSensor

__all__ = [
    "VibrationSensor",
    "TemperatureSensor",
    "CurrentSensor",
    "TorqueSensor",
    "TachometerSensor",
]
