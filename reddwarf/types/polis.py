from typing import TypeAlias, Literal
from typing_extensions import NotRequired, TypedDict

IncrementingId: TypeAlias = int
BaseClusterId: TypeAlias = IncrementingId
GroupId: TypeAlias = IncrementingId
ParticipantId: TypeAlias = IncrementingId

class PolisGroupCluster(TypedDict):
    id: GroupId
    members: list[BaseClusterId]
    center: list[float]

class PolisGroupClusterExpanded(TypedDict):
    id: GroupId
    members: list[ParticipantId]
    center: list[float]

class PolisBaseClusters(TypedDict):
    # Each outer list will be the same length, and will be 100 items or less.
    id: list[BaseClusterId]
    members: list[list[ParticipantId]]
    x: list[float]
    y: list[float]
    count: list[int]

# Use functional form when attributes have hyphens or are string numbers.
PolisRepnessStatement = TypedDict("PolisRepnessStatement", {
    "tid": int,
    "n-success": int,
    "n-trials": int,
    "p-success": float,
    "p-test": float,
    "repness": float,
    "repness-test": float,
    "repful-for": Literal["agree", "disagree"],
    "best-agree": NotRequired[bool],
    "n-agree": NotRequired[int],
})

PolisRepness = TypedDict("PolisRepness", {
    "0": list[PolisRepnessStatement],
    "1": list[PolisRepnessStatement],
    "2": NotRequired[list[PolisRepnessStatement]],
    "3": NotRequired[list[PolisRepnessStatement]],
    "4": NotRequired[list[PolisRepnessStatement]],
})