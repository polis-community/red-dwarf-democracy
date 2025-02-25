from typing import TypedDict, List, Optional, TypeAlias
from enum import IntEnum


Identifier: TypeAlias = int | str

class Statement(TypedDict):
    """
    Attributes:
        id (Identifier): Statement ID
    """
    id: Identifier

class Participant(TypedDict):
    """
    Attributes:
        id (Identifier): Participant ID
    """
    id: Identifier

class ClusteredParticipant(TypedDict):
    """
    Attributes:
        id (Identifier): Participant ID
        x (float): X coordinate
        y (float): Y coordinate
    """
    id: Identifier # participant.id
    x: float
    y: float

class VoteValueEnum(IntEnum):
    AGREE = 1
    DISAGREE = -1
    # Can withhold using "pass" at own discretion.
    PASS = 0

class Vote(TypedDict):
    """
    Attributes:
        statement_id (Identifier): Statement ID
        participant_id (Identifier): Participant ID
        vote (VoteValueEnum): Vote value
    """
    statement_id: Identifier # statement.id
    participant_id: Identifier # participant.id

    vote: VoteValueEnum

class Conversation(TypedDict):
    """
    Attributes:
        votes (list[Vote]): A list of votes
    """
    votes: List[Vote]

class Cluster(TypedDict):
    """
    Attributes:
        id (int): Cluster ID
        participants (list[ClusteredParticipant]): List of clustered participants.
    """
    id: int
    participants: List[ClusteredParticipant]

class ClusteringResult(TypedDict):
    """
    Attributes:
        clusters (list[Cluster]): List of clusters.
    """
    clusters: List[Cluster]

class ClusteringOptions(TypedDict):
    """
    Attributes:
        min_user_vote_threshold (Optional[int]): By default we filter out participants who've placed less than 7 votes. This overrides that.
        max_clusters (Optional[int]): By default we check kmeans for 2-5 groups. This overrides the upper bound.
    """
    min_user_vote_threshold: Optional[int]
    max_clusters: Optional[int]

def run_clustering(conversation: Conversation, options: ClusteringOptions) -> ClusteringResult:
    raise NotImplementedError
