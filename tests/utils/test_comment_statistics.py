from reddwarf import utils
from tests.fixtures import small_convo_math_data
import numpy as np
import pytest

from reddwarf.polis import PolisClient
from reddwarf.types.polis import PolisRepness

def get_grouped_statement_ids(repness: PolisRepness) -> dict[str, list[dict[str, list[int]]]]:
    groups = []

    for key, statements in repness.items():
        group = {"id": str(key), "members": [stmt["tid"] for stmt in statements]} # type:ignore
        groups.append(group)

    return {"groups": groups}

def test_calculate_representativeness_real_data(small_convo_math_data):
    math_data, path, _ = small_convo_math_data
    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepaths=[
        f'{path}/votes.json',
        f'{path}/comments.json',
    ])

    # Get group-clusters, but with members as participant ids rather than base cluster ids
    group_clusters_with_pids = utils.expand_group_clusters_with_participants(
        group_clusters=math_data["group-clusters"],
        base_clusters=math_data["base-clusters"],
    )

    # Get list of all active participant ids, since Polis has some edge-cases
    # that keep specific participants, and we need to keep them from being filtered out.
    all_participant_ids = utils.get_all_participant_ids(group_clusters_with_pids)
    client.keep_participant_ids = all_participant_ids
    vote_matrix = client.get_matrix(is_filtered=True)

    # Get sorted list of cluster labels from
    cluster_labels = utils.generate_cluster_labels(group_clusters_with_pids)

    # Generate stats all groups and all statements.
    stats_by_group = utils.calculate_comment_statistics_by_group(
        vote_matrix=vote_matrix,
        cluster_labels=np.array(cluster_labels),
    )

    polis_repness = utils.select_rep_comments(stats_by_group=stats_by_group)
    actual_repness: PolisRepness = polis_repness # type:ignore
    expected_repness: PolisRepness = math_data["repness"] # type:ignore
    assert get_grouped_statement_ids(actual_repness) == get_grouped_statement_ids(expected_repness)

    # Cycle through all the expected data calculated by Polis platform
    for group_id, statements in math_data['repness'].items():
        group_id = int(group_id)
        for st in statements:
            expected_repr = st["repness"]
            expected_repr_test = st["repness-test"]
            expected_prob = st["p-success"]
            expected_prob_test = st["p-test"]

            # Fetch matching calculated values for comparison.
            keys = ["prob", "prob_test", "repness", "repness_test"]
            if st["repful-for"] == "agree":
                key_map = dict(zip(keys, ["pa", "pat", "ra", "rat"]))
            elif st["repful-for"] == "disagree":
                key_map = dict(zip(keys, ["pd", "pdt", "rd", "rdt"]))

            actual = {
                k: stats_by_group[group_id].loc[st["tid"], v]
                for k,v in key_map.items()
            }

            assert actual["prob"] == pytest.approx(expected_prob)
            assert actual["prob_test"] == pytest.approx(expected_prob_test)
            assert actual["repness"] == pytest.approx(expected_repr)
            assert actual["repness_test"] == pytest.approx(expected_repr_test)
