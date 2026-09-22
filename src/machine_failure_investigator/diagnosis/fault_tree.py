"""Simple fault-tree structure for investigation reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FaultTreeNode:
    name: str
    children: List["FaultTreeNode"] = field(default_factory=list)
    description: str = ""


def default_fault_tree() -> FaultTreeNode:
    return FaultTreeNode(
        name="Abnormal condition",
        children=[
            FaultTreeNode(
                name="Rotational",
                children=[
                    FaultTreeNode("IMBALANCE", description="Elevated 1× component"),
                    FaultTreeNode("MISALIGNMENT", description="Elevated 2× / 3× harmonics"),
                    FaultTreeNode("SHAFT_DEFECT", description="Shaft-related 1×/2×"),
                ],
            ),
            FaultTreeNode(
                name="Bearing-related",
                children=[
                    FaultTreeNode("BEARING_DEGRADATION", description="HF energy, envelope peaks"),
                    FaultTreeNode("BEARING_FAILURE", description="Severe HF + temperature"),
                    FaultTreeNode("LUBRICATION_FAILURE", description="Friction + heat"),
                ],
            ),
            FaultTreeNode(
                name="Structural",
                children=[
                    FaultTreeNode("LOOSENESS", description="Half-order / nonlinear"),
                ],
            ),
            FaultTreeNode(
                name="Thermal / other",
                children=[
                    FaultTreeNode("OVERHEATING", description="Temperature-dominant"),
                    FaultTreeNode("GEAR_WEAR", description="Gear mesh and sidebands"),
                ],
            ),
            FaultTreeNode("UNKNOWN_ANOMALY", description="Out-of-distribution anomaly"),
        ],
    )


def path_to_fault(tree: FaultTreeNode, fault_name: str) -> Optional[List[str]]:
    def dfs(node: FaultTreeNode, path: List[str]) -> Optional[List[str]]:
        new_path = path + [node.name]
        if node.name == fault_name:
            return new_path
        for child in node.children:
            found = dfs(child, new_path)
            if found:
                return found
        return None

    return dfs(tree, [])
