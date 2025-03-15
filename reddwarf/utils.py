import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import Any, List, Dict, Tuple, Optional, Literal, TypeAlias
from reddwarf.exceptions import RedDwarfError
from scipy.stats import norm

from numpy.typing import ArrayLike
from types import SimpleNamespace

from reddwarf.types.polis import (
    PolisBaseClusters,
    PolisGroupCluster,
    PolisGroupClusterExpanded,
    GroupId,
    ParticipantId,
)

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
        cluster_labels, _ = run_kmeans(
            dataframe=projected_data,
            n_clusters=k_test,
            random_state=random_state,
        )
        this_silhouette_score = silhouette_score(projected_data, cluster_labels)
        if debug:
            print(f"{k_test=}, {this_silhouette_score=}")
        if this_silhouette_score >= best_silhouette_score:
            k_best = k_test
            best_silhouette_score = this_silhouette_score
            best_cluster_labels = cluster_labels

    optimal_k = k_best
    optimal_silhouette = best_silhouette_score
    optimal_cluster_labels = best_cluster_labels

    return optimal_k, optimal_silhouette, optimal_cluster_labels

def one_prop_test(
    succ: ArrayLike,
    n: ArrayLike,
) -> np.float64 | np.ndarray[np.float64]:
    # Convert inputs to numpy arrays (if they aren't already)
    succ = np.asarray(succ) + 1
    n = np.asarray(n) + 1

    # Compute the test statistic
    return 2 * np.sqrt(n) * ((succ / n) - 0.5)

# Calculate representativeness two-prop test
def two_prop_test(
    succ_in: ArrayLike,
    succ_out: ArrayLike,
    n_in: ArrayLike,
    n_out: ArrayLike,
) -> np.float64 | np.ndarray[np.float64]:
    """Two-prop test ported from Polis. Accepts numpy arrays for bulk processing."""
    # Ported with adaptation from Polis
    # See: https://github.com/compdemocracy/polis/blob/90bcb43e67dad660629e0888fedc0d32379f375d/math/src/polismath/math/stats.clj#L18-L33

    # Laplace smoothing (add 1 to each count)
    succ_in = np.asarray(succ_in) + 1
    succ_out = np.asarray(succ_out) +  1
    n_in = np.asarray(n_in) +  1
    n_out = np.asarray(n_out) +  1

    # Compute proportions
    pi1 = succ_in / n_in
    pi2 = succ_out / n_out
    pi_hat = (succ_in + succ_out) / (n_in + n_out)

    # Handle edge case when pi_hat == 1
    # TODO: Handle when divide by zero.
    if np.any(pi_hat == 1):
        # XXX: Not technically correct; limit-based solution needed.
        return np.where(pi_hat == 1, 0, (pi1 - pi2) / np.sqrt(pi_hat * (1 - pi_hat) * (1 / n_in + 1 / n_out)))

    # Compute the test statistic
    denominator = np.sqrt(pi_hat * (1 - pi_hat) * (1 / n_in + 1 / n_out))

    return (pi1 - pi2) / denominator

def z_sig_90(z_val) -> bool:
    """Test whether z-statistic is significant at 90% confidence (one-tailed, right-side)."""
    critical_value = norm.ppf(0.90)  # 90% confidence level, one-tailed
    return z_val > critical_value

def is_passes_by_test(pat, rat, pdt, rdt) -> bool:
    "Decide whether we should count a statement in a group as being representative."
    is_agreement_significant = z_sig_90(pat) and z_sig_90(rat)
    is_disagreement_significant = z_sig_90(pdt) and z_sig_90(rdt)

    return is_agreement_significant or is_disagreement_significant

def beats_best_by_test(rad, rdt, current_best) -> bool:
    """
    Returns True if a given comment/group stat has a more representative z-score than
    current_best_z. Used for ensuring at least one representative comment for every group,
    even if none remain after more thorough filters.
    """
    if current_best is not None:
        this_repness_test = max(rad, rdt)
        current_best_repness_test = max(current_best["rat"], current_best["rdt"])

        return this_repness_test > current_best_repness_test
    else:
        return True


def beats_best_agr(na, nd, ra, rat, pa, pat, ns, current_best) -> bool:
    """
    Like beats_best_by_test, but only considers agrees. Additionally, doesn't focus solely on repness,
    but also on raw probability of agreement, so as to ensure that there is some representation of what
    people in the group agree on. Also, note that this takes the current_best statement, instead of just current_best_z.
    """

    # Explicitly don't allow something that hasn't been voted on at all
    if na == 0 and nd == 0:
        return False

    # If we have a current_best by representativeness estimate, use the more robust measurement
    if (current_best is not None) and current_best["ra"] > 1.0:
        return (ra * rat * pa * pat) > (current_best["ra"] * current_best["rat"] * current_best["pa"] * current_best["pat"])

    # If we have current_best, but only by probability estimate, just shoot for something generally agreed upon
    if current_best is not None:
        return (pa * pat) > (current_best["pa"] * current_best["pat"])

    # Otherwise, accept if either representativeness or probability look generally good
    return z_sig_90(pat) or (ra > 1.0 and pa > 0.5)

# For making it more legible to get index of agree/disagree votes in numpy array.
votes = SimpleNamespace(A=0, D=1)

def count_votes(
    values: ArrayLike,
    vote_value: Optional[int] = None,
) -> np.int64 | np.ndarray[np.int64]:
    values = np.asarray(values)
    if vote_value:
        # Count votes that match value.
        return np.sum(values == vote_value, axis=0)
    else:
        # Count any non-missing values.
        return np.sum(np.isfinite(values), axis=0)

def count_disagree(values: ArrayLike) -> np.int64 | np.ndarray[np.int64]:
    return count_votes(values, -1)

def count_agree(values: ArrayLike) -> np.int64 | np.ndarray[np.int64]:
    return count_votes(values, 1)

def count_all_votes(values: ArrayLike) -> np.int64 | np.ndarray[np.int64]:
    return count_votes(values)

def probability(count, total, pseudo_count=1):
    """Probability with Laplace smoothing"""
    return (pseudo_count + count ) / (2*pseudo_count + total)

def calculate_comment_statistics(
    vote_matrix: VoteMatrix,
    cluster_labels: list[int],
    pseudo_count: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cluster_labels = np.asarray(cluster_labels)
    # Get the vote matrix values
    X = vote_matrix.values

    group_count = cluster_labels.max()+1
    statement_ids = vote_matrix.columns

    # Set up all the variables to be populated.
    N_g_c = np.empty([group_count, len(statement_ids)], dtype='int32')
    N_v_g_c = np.empty([len(votes.__dict__), group_count, len(statement_ids)], dtype='int32')
    P_v_g_c = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    R_v_g_c = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    P_v_g_c_test = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    R_v_g_c_test = np.empty([len(votes.__dict__), group_count, len(statement_ids)])

    for gid in range(group_count):
        # Create mask for the participants in target group
        in_group_mask = (cluster_labels == gid)
        X_in_group = X[in_group_mask]

        # Count any votes [-1, 0, 1] for all statements/features at once

        # NON-GROUP STATS

        # For in-group
        n_agree_in_group    = N_v_g_c[votes.A, gid, :] = count_agree(X_in_group)    # na
        n_disagree_in_group = N_v_g_c[votes.D, gid, :] = count_disagree(X_in_group) # nd
        n_votes_in_group    = N_g_c[gid, :] = count_all_votes(X_in_group)           # ns

        # Calculate probabilities
        p_agree_in_group    = P_v_g_c[votes.A, gid, :] = probability(n_agree_in_group, n_votes_in_group)    # pa
        p_disagree_in_group = P_v_g_c[votes.D, gid, :] = probability(n_disagree_in_group, n_votes_in_group) # pd

        # Calculate probability test z-scores
        P_v_g_c_test[votes.A, gid, :] = one_prop_test(n_agree_in_group, n_votes_in_group)    # pat
        P_v_g_c_test[votes.D, gid, :] = one_prop_test(n_disagree_in_group, n_votes_in_group) # pdt

        # GROUP COMPARISON STATS

        out_group_mask = ~in_group_mask
        X_out_group = X[out_group_mask]

        # For out-group
        n_agree_out_group = count_agree(X_out_group)
        n_disagree_out_group = count_disagree(X_out_group)
        n_votes_out_group = count_all_votes(X_out_group)

        # Calculate out-group probabilities
        p_agree_out_group    = probability(n_agree_out_group, n_votes_out_group)
        p_disagree_out_group = probability(n_disagree_out_group, n_votes_out_group)

        # Calculate representativeness
        R_v_g_c[votes.A, gid, :] = p_agree_in_group / p_agree_out_group       # ra
        R_v_g_c[votes.D, gid, :] = p_disagree_in_group / p_disagree_out_group # rd

        # Calculate representativeness test z-scores
        R_v_g_c_test[votes.A, gid, :] = two_prop_test(n_agree_in_group, n_agree_out_group, n_votes_in_group, n_votes_out_group)       # rat
        R_v_g_c_test[votes.D, gid, :] = two_prop_test(n_disagree_in_group, n_disagree_out_group, n_votes_in_group, n_votes_out_group) # rdt

    return (
        N_g_c, # ns
        N_v_g_c, # na / nd
        P_v_g_c, # pa / pd
        R_v_g_c, # ra / rd
        P_v_g_c_test, # pat / pdt
        R_v_g_c_test, # rat / rdt
    )

def finalize_cmt_stats(cmt: pd.Series) -> dict:
    if cmt["rat"] > cmt["rdt"]:
        vals = [cmt[k] for k in ["na", "ns", "pa", "pat", "ra", "rat"]] + ["agree"]
    else:
        vals = [cmt[k] for k in ["nd", "ns", "pd", "pdt", "rd", "rdt"]] + ["disagree"]

    return {
        "tid":          int(cmt["statement_id"]),
        "n-success":    int(vals[0]),
        "n-trials":     int(vals[1]),
        "p-success":    float(vals[2]),
        "p-test":       float(vals[3]),
        "repness":      float(vals[4]),
        "repness-test": float(vals[5]),
        "repful-for":   vals[6],
    }

def calculate_comment_statistics_by_group(
    vote_matrix: VoteMatrix,
    cluster_labels: list[int],
    pseudo_count: int = 1,
) -> list[pd.DataFrame]:
    """
    Calculate the Polis representativeness metric for every statement for a specific group.

    The representativeness metric is defined as:
    R_v(g,c) = P_v(g,c) / P_v(~g,c)

    Where:
    - P_v(g,c) is probability of vote v on comment c in group g
    - P_v(~g,c) is probability of vote v on comment c in all groups except g

    Args:
        vote_matrix (VoteMatrix): The vote matrix where rows are voters, columns are statements,
                                  and values are votes (1 for agree, -1 for disagree, 0 for pass).
        cluster_labels (np.ndarray): Array of cluster labels for each participant in the vote matrix.
        group_id (int): The ID of the group/cluster to calculate representativeness for.
        pseudo_count (int): Smoothing parameter to avoid division by zero. Default is 1.

    Returns:
        representativeness (pd.DataFrame): DataFrame containing representativeness scores for each comment,
                                          with columns for agree_repr, disagree_repr, and n_votes.
    """
    cluster_labels = np.asarray(cluster_labels)
    N_g_c, N_v_g_c, P_v_g_c, R_v_g_c, P_v_g_c_test, R_v_g_c_test = calculate_comment_statistics(
        vote_matrix=vote_matrix,
        cluster_labels=cluster_labels,
        pseudo_count=pseudo_count,
    )

    group_count = cluster_labels.max()+1
    group_stats = [None] * group_count
    for group_id in range(group_count):
        group_stats[group_id] = pd.DataFrame({
            'na':  N_v_g_c[votes.A, group_id, :],
            'nd':  N_v_g_c[votes.D, group_id, :],
            'ns':  N_g_c[group_id, :],
            'pa':  P_v_g_c[votes.A, group_id, :],
            'pd':  P_v_g_c[votes.D, group_id, :],
            'pat': P_v_g_c_test[votes.A, group_id, :],
            'pdt': P_v_g_c_test[votes.D, group_id, :],
            'ra':  R_v_g_c[votes.A, group_id, :],
            'rd':  R_v_g_c[votes.D, group_id, :],
            'rat': R_v_g_c_test[votes.A, group_id, :],
            'rdt': R_v_g_c_test[votes.D, group_id, :],
        }, index=vote_matrix.columns)

    return group_stats

# Figuring out select-rep-comments flow
# See: https://github.com/compdemocracy/polis/blob/7bf9eccc287586e51d96fdf519ae6da98e0f4a70/math/src/polismath/math/repness.clj#L209C7-L209C26
# TODO: omg please clean this up.
def select_rep_comments(stats_by_group: list[pd.DataFrame], pick_n: int = 5):
    polis_repness = {}
    for gid, stats_df in enumerate(stats_by_group):
        sufficient_statements = []
        best_agree_of_sufficient = None
        best_of_all = None

        def create_filter_mask(row):
            return is_passes_by_test(row["pat"], row["rat"], row["pdt"], row["rdt"])

        sufficient_statements = stats_df[stats_df.apply(create_filter_mask, axis=1)]
        sufficient_statements = pd.DataFrame([
                finalize_cmt_stats(row)
                for _, row in sufficient_statements.reset_index().iterrows()
            ],
            index=sufficient_statements.index,
        )

        if len(sufficient_statements) > 0:
            repness_metric = lambda row: row["repness"] * row["repness-test"] * row["p-success"] * row["p-test"]
            sufficient_statements = (sufficient_statements
                .assign(sort_order=repness_metric)
                .sort_values(by="sort_order", ascending=False)
            )

        # Track the best, even if doesn't meet sufficient minimum, to have at least one.
        # TODO: Merge this wil above iteration
        if len(sufficient_statements) == 0:
            for _, row in stats_df.reset_index().iterrows():
                if beats_best_by_test(row["rat"], row["rdt"], best_of_all):
                    best_of_all = row

        # Track the best-agree, to bring to top if exists.
        for _, row in stats_df.reset_index().iterrows():
            if beats_best_agr(row["na"], row["nd"], row["ra"], row["rat"], row["pa"], row["pat"], row["ns"], best_agree_of_sufficient):
                best_agree_of_sufficient = row

        # Start building repness key
        if best_agree_of_sufficient is not None:
            best_agree_of_sufficient = finalize_cmt_stats(best_agree_of_sufficient)
            best_agree_of_sufficient.update({"n-agree": best_agree_of_sufficient["n-success"], "best-agree": True})
            best_head = [best_agree_of_sufficient]
        elif best_of_all is not None:
            best_head = [best_of_all]
        else:
            best_head = []

        if len(sufficient_statements) > 0:
            sufficient_statements = sufficient_statements.drop(columns="sort_order")
        if len(best_head) > 0:
            selected = best_head + [dict(row) for _, row in sufficient_statements.iterrows() if row["tid"] != best_head[0]["tid"]]
        else:
            selected = [dict(row) for _, row in sufficient_statements.iterrows()]
        # sorted() does the work of agrees-before-disagrees in polismath
        polis_repness[str(gid)] = sorted(selected[:pick_n], key=lambda x: x["repful-for"])

    return polis_repness

## POLISMATH HELPERS
#
# These functions are generally used to transform data from the polismath object
# that gets returned from the /api/v3/math/pca2 endpoint.

def expand_group_clusters_with_participants(
    group_clusters: list[PolisGroupCluster],
    base_clusters: PolisBaseClusters,
) -> list[PolisGroupClusterExpanded]:
    """
    Expand group clusters to include direct participant IDs instead of base cluster IDs.

    Args:
        group_clusters (list[PolisGroupCluster]): Group cluster data from Polis math data, with base cluster ID membership.
        base_clusters (PolisBaseClusters): Base cluster data from Polis math data, with participant ID membership.

    Returns:
        group_clusters_with_participant_ids (list[PolisGroupClusterExpanded]): A list of group clusters with participant IDs.
    """
    # Extract base clusters and their members
    base_cluster_ids = base_clusters["id"]
    base_cluster_participant_ids = base_clusters["members"]

    # Create a mapping from base cluster ID to member participant IDs
    base_to_participant_mapping = dict(zip(base_cluster_ids, base_cluster_participant_ids))

    # Create the new group-clusters-participant list
    group_clusters_with_participant_ids = []

    for group in group_clusters:
        # Get all participant IDs for this group by expanding and flattening the base cluster members.
        group_participant_ids = [
            participant_id
            for base_cluster_id in group["members"]
            for participant_id in base_to_participant_mapping[base_cluster_id]
        ]

        # Create the new group object with expanded participant IDs
        expanded_group = {
            **group,
            "id": group["id"],
            "members": group_participant_ids,
        }

        group_clusters_with_participant_ids.append(expanded_group)

    return group_clusters_with_participant_ids

def generate_cluster_labels(
    group_clusters_with_pids: list[PolisGroupClusterExpanded],
) -> np.ndarray[GroupId]:
    """
    Transform group_clusters_with_pid into an ordered list of cluster IDs
    sorted by participant ID, suitable for cluster labels.

    Args:
        group_clusters_with_pids (list[PolisGroupClusterExpanded]): List of group clusters with expanded participant IDs

    Returns:
        np.ndarray[GroupId]: A list of cluster IDs ordered by participant ID
    """
    # Create a list of (participant_id, cluster_id) pairs
    participant_cluster_pairs = []

    for cluster in group_clusters_with_pids:
        cluster_id = cluster["id"]
        for participant_id in cluster["members"]:
            participant_cluster_pairs.append((participant_id, cluster_id))

    # Sort the list by participant_id
    sorted_pairs = sorted(participant_cluster_pairs, key=lambda x: x[0])

    # Extract just the cluster IDs in order
    cluster_ids_in_order = [pair[1] for pair in sorted_pairs]

    return np.asarray(cluster_ids_in_order)

def get_all_participant_ids(
    group_clusters_with_pids: list[PolisGroupClusterExpanded],
) -> list[ParticipantId]:
    """
    Extract all unique participant IDs from group_clusters_with_pids.

    Args:
        group_clusters_with_pids (list): List of group clusters with expanded participant IDs

    Returns:
        list: A sorted list of all unique participant IDs
    """
    # Gather all participant IDs from all clusters
    all_participant_ids = []

    for cluster in group_clusters_with_pids:
        all_participant_ids.extend(cluster["members"])

    # Remove duplicates and sort
    unique_participant_ids = sorted(set(all_participant_ids))

    return unique_participant_ids

def extract_data_from_polismath(math_data: Any):
    """
    A helper to extract specific types of data from polismath data, to avoid having to recalculate it.

    This is mostly helpful for debugging and working with the existing Polis platform API.

    Args:
        math_data (Any): The polismath data object that is returned from /api/v3/math/pca2

    Returns:
        list[ParticipantId]: A sorted list of all clustered/grouped Polis participant IDs
        np.ndarray[GroupId]: A list of all cluster labels (aka group IDs), sorted by participant ID
    """
    group_clusters_with_pids = expand_group_clusters_with_participants(
        group_clusters=math_data["group-clusters"],
        base_clusters=math_data["base-clusters"],
    )

    grouped_participant_ids = get_all_participant_ids(group_clusters_with_pids)
    cluster_labels = generate_cluster_labels(group_clusters_with_pids)

    return grouped_participant_ids, cluster_labels