from typing import Annotated
from reddwarf.types.demdis import ClusteringResult, VoteModel, ClusteringCenterModel, ClusteringCenter, VoteValueEnum as DemDisVoteValueEnum
from reddwarf.types.base import VoteValueEnum as BaseVoteValueEnum
from reddwarf import utils
from datetime import datetime


DEFAULT_MIN_USER_VOTE_THRESHOLD = 7
DEFAULT_MAX_CLUSTERS = 5
DEFAULT_CLUSTER_NAMES = ["A", "B", "C", "D", "E"]

DEMDIS_VOTE_KEY_MAPPING = {
    "voted_by_participant_id": "participant_id",
    "value": "vote",
}

DEMDIS_VOTE_VALUE_MAPPING = {
    DemDisVoteValueEnum.AGREE:    BaseVoteValueEnum.UP,
    DemDisVoteValueEnum.SKIP:     BaseVoteValueEnum.NEUTRAL,
    DemDisVoteValueEnum.DISAGREE: BaseVoteValueEnum.DOWN,
}

def remap_vote_values(votes, value_mapping, key_mapping = {}):
    rekeyed_votes = [
        {
            # Use key_mapping if available, otherwise keep key unchanged.
            (key_mapping.get(k, k)): val
                for k, val in vote.items()
        }
        for vote in votes
    ]

    remapped_votes = [
        {
            # Use vote value_mapping if available, otherwise keep value unchanged.
            k: (value_mapping.get(val, val) if k == "vote" else val)
                for k, val in vote.items()
        }
        for vote in rekeyed_votes
    ]

    return remapped_votes

# See: https://github.com/Demdis/Clustering-types/blob/main/types.py
def run_clustering(
    *,
    votes: list[VoteModel],
    reference_cluster_centers: list[ClusteringCenterModel] | None,
    statement_boost: tuple[Annotated[int, "statement id"], Annotated[float, "boost"]] | None = None,
    specific_cluster_count: int | None = None,
    skip_remap = False,
) -> ClusteringResult:
    if not skip_remap:
        votes = remap_vote_values(votes, DEMDIS_VOTE_VALUE_MAPPING, DEMDIS_VOTE_KEY_MAPPING)

    last_vote_timestamp = 0
    for vote in votes:
        if vote["modified"] > last_vote_timestamp:
            last_vote_timestamp = vote["modified"]

    raw_vote_matrix = utils.generate_raw_matrix(votes=votes)
    all_statement_ids = raw_vote_matrix.columns

    filtered_vote_matrix = utils.filter_matrix(
        vote_matrix=raw_vote_matrix,
        min_user_vote_threshold=DEFAULT_MIN_USER_VOTE_THRESHOLD,
        active_statement_ids=all_statement_ids,
    )

    projected_data, _, _ = utils.run_pca(vote_matrix=filtered_vote_matrix)

    projected_data = utils.scale_projected_data(
        projected_data=projected_data,
        vote_matrix=filtered_vote_matrix,
    )

    # TODO: Confirm init_centers works.
    if specific_cluster_count:
        cluster_labels, cluster_centers = utils.run_kmeans(
            dataframe=projected_data,
            n_clusters=specific_cluster_count,
            init_centers=([[c["center_x"], c["center_y"]] for c in reference_cluster_centers] if reference_cluster_centers else None)
        )
    else:
        _, _, cluster_labels, cluster_centers = utils.find_optimal_k(
            projected_data=projected_data,
            max_group_count=DEFAULT_MAX_CLUSTERS,
            init_centers=([[c["center_x"], c["center_y"]] for c in reference_cluster_centers] if reference_cluster_centers else None)
        )

    # Add cluster label column to dataframe.
    projected_data = projected_data.assign(cluster_id=cluster_labels)

    merged_df = projected_data.join(filtered_vote_matrix)


    def build_centers(df):
        centers = [
            {
                "name": DEFAULT_CLUSTER_NAMES[cluster_id],
                "center_x": float(cluster_centers[cluster_id][0]),
                "center_y": float(cluster_centers[cluster_id][1]),
                "participant_count": (group_participant_count := len(group_df)),
                "participants": [
                    {
                        "participant_id": row.index,
                        "cluster_center_name": DEFAULT_CLUSTER_NAMES[cluster_id],
                        "x": row.x,
                        "y": row.y,
                    }
                    for row in group_df.itertuples(index=True)
                ],
                "statements": [
                    {
                        "statement_id": statement_id,
                        "cluster_center_name": DEFAULT_CLUSTER_NAMES[cluster_id],
                        "agreement_count": (agreement_count := int((group_df[statement_id] == 1).sum())),
                        "disagreement_count": int((group_df[statement_id] == -1).sum()),
                        "skip_count": int((group_df[statement_id] == 0).sum()),
                        "unseen_count": int(group_df[statement_id].isna().sum()),
                        "agreement_percentage": (agreement_count / group_participant_count) * 100,
                        "cluster_defining_pos_coefficient": 0.0, # TODO
                        "cluster_defining_neg_coefficient": 0.0, # TODO
                        "cluster_defining_skip_coefficient": 0.0, # TODO
                    }
                    for statement_id in filtered_vote_matrix.columns
                ],
            }
            for cluster_id, group_df in df.groupby("cluster_id")
        ]

        return centers

    result: ClusteringResult = {
        "participant_count": len(raw_vote_matrix.index),
        "participants_clustered": len(filtered_vote_matrix.index),
        "vote_count": int(raw_vote_matrix.count().sum()),
        "statement_count": len(raw_vote_matrix.columns),
        "last_vote_at": datetime.fromtimestamp(last_vote_timestamp/1000),

        "statement_metrics": [
            {
                "statement_id": statement_id,
                "mean_agreement_percentage": 0.0, # TODO
                "consensus_points": 0, # TODO
                "polarization_measurement": 0.0, # TODO

            }
            for statement_id in filtered_vote_matrix.columns
        ],
        "centers": build_centers(merged_df),
    }

    return result
