from typing import Annotated
from datetime import datetime


# Source: https://github.com/Demdis/Clustering-types/blob/main/types.py

class VoteValueEnum(str):
    AGREE = "agree"
    DISAGREE = "disagree"
    SKIP = "skip"


class VoteModel():
    id: int
    conversation_id: int
    statement_id: int
    voted_by_participant_id: int

    value: VoteValueEnum


class ClusteringCenterModel():
    center_x: float
    center_y: float


class StatementConversationMetric():
    statement_id: int
    mean_agreement_percentage: float
    consensus_points: int
    polarization_measurement: float


class ClusteringParticipant():
    participant_id: int
    cluster_center_name: str
    x: float
    y: float


class ClusteredStatement():
    statement_id: int
    cluster_center_name: str
    agreement_count: int
    disagreement_count: int
    skip_count: int
    unseen_count: int
    agreement_percentage: float
    cluster_defining_pos_coefficient: float
    cluster_defining_neg_coefficient: float
    cluster_defining_skip_coefficient: float


class ClusteringCenter():
    name: str
    center_x: float
    center_y: float
    participant_count: int

    participants: list[ClusteringParticipant]
    statements: list[ClusteredStatement]


class ClusteringResult():
    participant_count: int
    participants_clustered: int
    vote_count: int
    statement_count: int
    last_vote_at: datetime

    statement_metrics: list[StatementConversationMetric]
    centers: list[ClusteringCenter]
