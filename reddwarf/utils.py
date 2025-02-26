import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Literal, TypeAlias
from reddwarf.exceptions import RedDwarfError

VoteMatrix: TypeAlias = pd.DataFrame

# TODO: Extract utils methods into individual modules.

def impute_missing_votes(vote_matrix: VoteMatrix) -> VoteMatrix:
    """
    Imputes missing votes in a voting matrix using column-wise mean. All columns must have at least one vote.

    Reference:
        Small, C. (2021). "Polis: Scaling Deliberation by Mapping High Dimensional Opinion Spaces."
        Specific highlight: <https://hyp.is/8zUyWM5fEe-uIO-J34vbkg/gwern.net/doc/sociology/2021-small.pdf>

    Args:
        vote_matrix (pd.DataFrame):  A vote matrix DataFrame with `NaN`/`None` values where: \
                                        1. rows are voters, \
                                        2. columns are statements, and \
                                        3. values are votes.

    Returns:
        imputed_matrix (pd.DataFrame): The same vote matrix DataFrame imputing missing values with column mean.
    """
    if vote_matrix.isna().all(axis="rows").any():
        raise RedDwarfError("impute_missing_votes does not support vote matrices containing statement columns with no votes.")

    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_matrix = pd.DataFrame(
        mean_imputer.fit_transform(vote_matrix),
        columns=vote_matrix.columns,
        index=vote_matrix.index,
    )
    return imputed_matrix

def filter_votes(
        votes: List[Dict],
        cutoff: Optional[int] = None,
) -> List[Dict]:
    """
    Filters a list of votes.

    If a `cutoff` is provided, votes are filtered based on either:

    - An `int` representing unix timestamp (ms), keeping only votes before or at that time.
        - Any int above 13_000_000_000 is considered a timestamp.
    - Any other positive or negative `int` is considered an index, reflecting where to trim the time-sorted vote list.
        - positive: filters in votes that many indices from start
        - negative: filters out votes that many indices from end

    Args:
        votes (List[Dict]): An unsorted list of vote records, where each record is a dictionary containing:

            - "participant_id": The ID of the voter.
            - "statement_id": The ID of the statement being voted on.
            - "vote": The recorded vote value.
            - "modified": A unix timestamp object representing when the vote was made.

        cutoff (int): A cutoff unix timestamp (ms) or index position in date-sorted votes list.

    Returns:
        votes (List[Dict]): An list of vote records, sorted by `modified` if index-based filtering occurred.
    """
    if cutoff:
        # TODO: Detect datetime object as arg instead.
        try:
            if cutoff > 1_300_000_000:
                cutoff_timestamp = cutoff
                votes = [v for v in votes if v['modified'] <= cutoff_timestamp]
            else:
                cutoff_index = cutoff
                votes = sorted(votes, key=lambda x: x["modified"])
                votes = votes[:cutoff_index]
        except KeyError as e:
            raise RedDwarfError("The `modified` key is missing from a vote object that must be sorted") from e

    return votes



def generate_raw_matrix(
        votes: List[Dict],
        cutoff: Optional[int] = None,
) -> VoteMatrix:
    """
    Generates a raw vote matrix from a list of vote records.

    See `filter_votes` method for details of `cutoff` arg.

    Args:
        votes (List[Dict]): An unsorted list of vote records, where each record is a dictionary containing:

            - "participant_id": The ID of the voter.
            - "statement_id": The ID of the statement being voted on.
            - "vote": The recorded vote value.
            - "modified": A unix timestamp object representing when the vote was made.

        cutoff (int): A cutoff unix timestamp (ms) or index position in date-sorted votes list.

    Returns:
        raw_matrix (pd.DataFrame): A full raw vote matrix DataFrame with NaN values where:

            1. rows are voters,
            2. columns are statements, and
            3. values are votes.

            This includes even voters that have no votes, and statements on which no votes were placed.
    """
    if cutoff:
        votes = filter_votes(votes=votes, cutoff=cutoff)

    raw_matrix = pd.DataFrame.from_dict(votes)
    raw_matrix = raw_matrix.pivot(
        values="vote",
        index="participant_id",
        columns="statement_id",
    )

    return raw_matrix

def get_unvoted_statement_ids(vote_matrix: VoteMatrix) -> List[int]:
    """
    A method intended to be piped into a VoteMatrix DataFrame, returning list of unvoted statement IDs.

    See: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html>

    Args:
        vote_matrix (pd.DataFrame): A pivot of statements (cols), participants (rows), with votes as values.

    Returns:
        unvoted_statement_ids (List[int]): list of statement IDs with no votes.

    Example:

        unused_statement_ids = vote_matrix.pipe(get_unvoted_statement_ids)
    """
    null_column_mask = vote_matrix.isnull().all()
    null_column_ids = vote_matrix.columns[null_column_mask].tolist()

    return null_column_ids

def filter_matrix(
        vote_matrix: VoteMatrix,
        min_user_vote_threshold: int = 7,
        active_statement_ids: List[int] = [],
        keep_participant_ids: List[int] = [],
        unvoted_filter_type: Literal["drop", "zero"] = "drop",
) -> VoteMatrix:
    """
    Generates a filtered vote matrix from a raw matrix and filter config.

    Args:
        vote_matrix (pd.DataFrame): The [raw] vote matrix.
        min_user_vote_threshold (int): The number of votes a participant must make to avoid being filtered.
        active_statement_ids (List[int]): The statement IDs that are not moderated out.
        keep_participant_ids (List[int]): Preserve specific participants even if below threshold.
        unvoted_filter_type ("drop" | "zero"): When a statement has no votes, it can't be imputed. \
            This determined whether to drop the statement column, or set all the value to zero/pass. (Default: drop)

    Returns:
        filtered_vote_matrix (VoteMatrix): A vote matrix with the following filtered out:

            1. statements without any votes,
            2. statements that have been moderated out,
            3. participants below the vote count threshold,
            4. participants who have not been explicitly selected to circumvent above filtering.
    """
    # Filter out moderated statements.
    vote_matrix = vote_matrix.filter(active_statement_ids, axis='columns')
    # Filter out participants with less than 7 votes (keeping IDs we're forced to)
    # Ref: https://hyp.is/JbNMus5gEe-cQpfc6eVIlg/gwern.net/doc/sociology/2021-small.pdf
    participant_ids_meeting_vote_thresh = vote_matrix[vote_matrix.count(axis="columns") >= min_user_vote_threshold].index.to_list()
    # Add in some specific participant IDs for Polismath edge-cases.
    # See: https://github.com/compdemocracy/polis/pull/1893#issuecomment-2654666421
    participant_ids_in = participant_ids_meeting_vote_thresh + keep_participant_ids
    participant_ids_in_unique = list(set(participant_ids_in))
    vote_matrix = (vote_matrix
        .filter(participant_ids_in_unique, axis='rows')
        # .filter() and .drop() lost the index name, so bring it back.
        .rename_axis("participant_id")
    )

    # This is otherwise the more efficient way, but we want to keep some participant IDs
    # to troubleshoot edge-cases in upsteam Polis math.
    # self.matrix = self.matrix.dropna(thresh=self.min_votes, axis='rows')

    unvoted_statement_ids = vote_matrix.pipe(get_unvoted_statement_ids)

    # TODO: What about statements with no votes? E.g., 53 in oprah. Filter out? zero?
    # Test this on a conversation where it will actually change statement count.
    if unvoted_filter_type == 'drop':
        vote_matrix = vote_matrix.drop(unvoted_statement_ids, axis='columns')
    elif unvoted_filter_type == 'zero':
        vote_matrix[unvoted_statement_ids] = 0

    return vote_matrix

def run_pca(
        vote_matrix: VoteMatrix,
        n_components: int = 2,
) -> Tuple[ pd.DataFrame, np.ndarray, np.ndarray ]:
    """
    Process a prepared vote matrix to be imputed and return projected participant data,
    as well as eigenvectors and eigenvalues.

    The vote matrix should not yet be imputed, as this will happen within the method.

    Args:
        vote_matrix (pd.DataFrame): A vote matrix of data. Non-imputed values are expected.
        n_components (int): Number n of principal components to decompose the `vote_matrix` into.

    Returns:
        projected_data (pd.DataFrame): A dataframe of projected xy coordinates for each `vote_matrix` row.
        eigenvectors (List[List[float]]): Principal `n` components, one per row.
        eigenvalues (List[float]): Explained variance,  one per row.
    """
    imputed_matrix = impute_missing_votes(vote_matrix)

    pca = PCA(n_components=n_components) ## pca is apparently different, it wants
    pca.fit(imputed_matrix) ## .T transposes the matrix (flips it)

    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_

    # Project participant vote data onto 2D using eigenvectors.
    projected_data = pca.transform(imputed_matrix)
    projected_data = pd.DataFrame(projected_data, index=imputed_matrix.index, columns=["x", "y"])
    projected_data.index.name = "participant_id"

    return projected_data, eigenvectors, eigenvalues

def scale_projected_data(
        projected_data: pd.DataFrame,
        vote_matrix: VoteMatrix
) -> pd.DataFrame:
    """
    Scale projected participant xy points based on vote matrix, to account for any small number of
    votes by a participant and prevent those participants from bunching up in the center.

    Args:
        projected_data (pd.DataFrame): the project xy coords of participants.
        vote_matrix (VoteMatrix): the processed vote matrix data frame, from which to generate scaling factors.

    Returns:
        scaled_projected_data (pd.DataFrame): The coord data rescaled based on participant votes.
    """
    total_active_comment_count = vote_matrix.shape[1]
    participant_vote_counts = vote_matrix.count(axis="columns")
    # Ref: https://hyp.is/x6nhItMMEe-v1KtYFgpOiA/gwern.net/doc/sociology/2021-small.pdf
    # Ref: https://github.com/compdemocracy/polis/blob/15aa65c9ca9e37ecf57e2786d7d81a4bd4ad37ef/math/src/polismath/math/pca.clj#L155-L156
    participant_scaling_coeffs = np.sqrt(total_active_comment_count / participant_vote_counts).values
    # See: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # Reshape scaling_coeffs list to match the shape of projected_data matrix
    participant_scaling_coeffs = np.reshape(participant_scaling_coeffs, (-1, 1))

    return projected_data * participant_scaling_coeffs

# TODO: Start passing init_centers based on /math/pca2 endpoint data,
# and see how often we get the same clusters.
def run_kmeans(
        dataframe: pd.DataFrame,
        n_clusters: int = 2,
        # TODO: Improve this type. 3d?
        init_centers: Optional[List] = None,
        random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs K-Means clustering on a 2D DataFrame of xy points, for a specific K,
    and returns labels for each row and cluster centers. Optionally accepts
    guesses on cluster centers, and a random_state to reproducibility.

    Args:
        dataframe (pd.DataFrame): A dataframe with two columns (assumed `x` and `y`).
        n_clusters (int): How many clusters k to assume.
        init_centers (List): A list of xy coordinates to use as initial center guesses.
        random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.

    Returns:
        cluster_labels (np.ndarray): A list of zero-indexed labels for each row in the dataframe
        cluster_centers (np.ndarray): A list of center coords for clusters.
    """
    if init_centers:
        # Pass an array of xy coords to see kmeans guesses.
        init_arg = init_centers[:n_clusters]
    else:
        # Use the default strategy in sklearn.
        init_arg = "k-means++"
    # TODO: Set random_state to a value eventually, so calculation is deterministic.
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init=init_arg,
        n_init="auto"
    ).fit(dataframe)

    return kmeans.labels_, kmeans.cluster_centers_

def find_optimal_k(
        projected_data: pd.DataFrame,
        max_group_count: int = 5,
        init_centers: Optional[List] = None,
        random_state: Optional[int] = None,
        debug: bool = False,
) -> Tuple[int, float, np.ndarray]:
    """
    Use silhouette scores to find the best number of clusters k to assume to fit the data.

    Args:
        projected_data (pd.DataFrame): A dataframe with two columns (assumed `x` and `y`).
        max_group_count (int): The max K number of groups to test for. (Default: 5)
        random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
        debug (bool): Whether to print debug output. (Default: False)

    Returns:
        optimal_k (int): Ideal number of clusters.
        optimal_silhouette_score (float): Silhouette score for this K value.
        optimal_cluster_labels (np.ndarray): A list of index labels assigned a group to each row in projected_date.
    """
    K_RANGE = range(2, max_group_count+1)
    k_best = 0 # Best K so far.
    best_silhouette_score = -np.inf

    for k_test in K_RANGE:
        cluster_labels, cluster_centers = run_kmeans(
            dataframe=projected_data,
            n_clusters=k_test,
            init_centers=init_centers,
            random_state=random_state,
        )
        this_silhouette_score = silhouette_score(projected_data, cluster_labels)
        if debug:
            print(f"{k_test=}, {this_silhouette_score=}")
        if this_silhouette_score >= best_silhouette_score:
            k_best = k_test
            best_silhouette_score = this_silhouette_score
            best_cluster_labels = cluster_labels
            best_cluster_centers = cluster_centers

    optimal_k = k_best
    optimal_silhouette = best_silhouette_score
    optimal_cluster_labels = best_cluster_labels
    optimal_cluster_centers = best_cluster_centers

    return optimal_k, optimal_silhouette, optimal_cluster_labels, optimal_cluster_centers
