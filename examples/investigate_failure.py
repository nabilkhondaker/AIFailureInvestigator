#!/usr/bin/env python3
"""Full investigation example with progressive bearing degradation."""

from machine_failure_investigator.diagnosis.investigator import Investigator
from machine_failure_investigator.reporting.report_builder import ReportBuilder
from machine_failure_investigator.simulation.scenarios import bearing_degradation_trajectory
from machine_failure_investigator.simulation.simulator import MachineSimulator

inv = Investigator.from_config("configs/system/default.yaml")
sim = MachineSimulator()
states = bearing_degradation_trajectory(n_windows=8, end_severity=0.9)

for state in states:
    bundle = sim.simulate(state, duration_s=0.5)
    result = inv.run(bundle)
    print(
        f"t={state.time_index:02d} sev={state.fault.severity:.2f} "
        f"→ {result.primary_hypothesis} conf={result.confidence:.2f} "
        f"anom={result.anomaly_detected}"
    )

# Final report
final_bundle = sim.simulate(states[-1])
final = inv.run(final_bundle)
print("\n" + ReportBuilder().build(final, machine_id="Trajectory Demo"))
