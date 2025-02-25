from reddwarf.types.agora import Conversation, ClusteringResult, ClusteringOptions
from reddwarf import utils

DEFAULT_MIN_USER_VOTE_THRESHOLD = 7
DEFAULT_MAX_CLUSTERS = 5
DEFAULT_KMEANS_RANDOM_STATE = 123456789

def run_clustering_v1(
    conversation: Conversation,
    options: ClusteringOptions = {},
) -> ClusteringResult:
    """
    A minimal Polis-based clustering agorithm suitable for use by Agora Citizen Network.

    This does the following:

    1. builds a vote matrix (includes as statement with at least 1 participant vote),
    2. filters out any participants with less than 7 votes,
    3. runs PCA and projects active participants into 2D coordinates,
    4. scales the projected participants out from center when low number of votes,
    5. test 2-5 groups for best k-means fit via silhouette scores (random state set for reproducibility)
    6. returns a list of clusters, each with a list of participant members and their projected 2D coordinates.

    Warning:
        This will technically function without PASS votes, but scaling
        factors will not be effective in compensating for missing votes,
        and so participant projections will be bunched up closer to the
        origin.

    Args:
        conversation (Conversation): A minimal conversation object with votes.
        options (ClusteringOptions): Configuration options for override defaults.

    Returns:
        result (ClusteringResult): Results of the clustering operation.
    """
    vote_matrix = utils.generate_raw_matrix(votes=conversation["votes"])
    # Any statements with votes are included.
    all_statement_ids = vote_matrix.columns
    vote_matrix = utils.filter_matrix(
        vote_matrix=vote_matrix,
        min_user_vote_threshold=options.get("min_user_vote_threshold", DEFAULT_MIN_USER_VOTE_THRESHOLD),
        active_statement_ids=all_statement_ids,
    )
    projected_data, _, _ = utils.run_pca(vote_matrix=vote_matrix)
    projected_data = utils.scale_projected_data(
        projected_data=projected_data,
        vote_matrix=vote_matrix,
    )

    _, _, cluster_labels = utils.find_optimal_k(
        projected_data=projected_data,
        max_group_count=options.get("max_clusters", DEFAULT_MAX_CLUSTERS),
        # Ensure reproducible kmeans calculation between runs.
        random_state=DEFAULT_KMEANS_RANDOM_STATE,
    )

    # Add cluster label column to dataframe.
    projected_data = projected_data.assign(cluster_id=cluster_labels)
    # Convert participant_id index into regular column, for ease of transformation.
    projected_data = projected_data.reset_index()

    result: ClusteringResult = {
        "clusters": [
            {
                "id": cluster_id,
                "participants": [
                    {
                        "id": row.participant_id,
                        "x": row.x,
                        "y": row.y,
                    }
                    for row in group.itertuples(index=False)
                ]
            }
            for cluster_id, group in projected_data.groupby("cluster_id")
        ]
    }

    return result
