from reddwarf.polis import PolisClient
from reddwarf.data_presenter import DataPresenter
import pandas as pd


CONVOS = {
    # Topic: What were the most significant developments in tech and politics in 2018?
    # 5 groups, 65 ptpts (56 grouped), 43 comments (open)
    "tech-politics-2018": {
        "report_id": "r2dfw8eambusb8buvecjt",
        "convo_id": "6jrufhr6dp",
    },
    # Topic: How should we operate vehicle-for-hire, e.g. Uber, Lyft and taxis in Toronto?
    # 2 groups, 47 ptpts (36 grouped), 69 comments (open)
    "rideshare-toronto": {
        "report_id": "r8xhmkwp6shm9yfermteh",
        "convo_id": "7vampckwrh",
    },
    # Topic: Help us pick rules for our AI chatbot! 7/7
    # 2 groups, 1_127 ptpts (1_094 grouped), 1_418 comments (closed)
    "anthropic-ccai": {
        "report_id": "r3rwrinr5udrzwkvxtdkj",
        "convo_id": "3akt5cdsfk",
    },
    # Topic: How should we use open source tools in governmment?
    # Test convo using xids and avatar images.
    "xid-testing": {
        "convo_id": "4kjz5rrrfe",
    },
}

if False:
    report_id = CONVOS["tech-politics-2018"]["report_id"]
    print(f"Loading data from report: https://pol.is/report/{report_id}")

    client = PolisClient()
    client.load_data(report_id=report_id)
    # To see the consequences of not having pass/neutral/zero votes
    DO_STRIP_PASS=False
    if DO_STRIP_PASS:
        client.votes = []
        client.load_votes_data(data=[v for v in client.data_loader.votes_data if v["vote"] != 0])
    client.get_matrix(is_filtered=True)
    client.run_pca()
    client.scale_projected_data()
    client.find_optimal_k()

    presenter = DataPresenter(client=client)
    presenter.render_optimal_cluster_figure()
    # client.generate_figure(coord_dataframe=client.projected_data)

if True:
    # test demdis method
    from reddwarf.demdis import run_clustering

    report_id = CONVOS["tech-politics-2018"]["report_id"]
    print(f"Loading data from report: https://pol.is/report/{report_id}")

    client = PolisClient()
    client.load_data(report_id=report_id)

    results = run_clustering(
        votes=client.data_loader.votes_data,
        reference_cluster_centers=None,
        # specific_cluster_count=2,
        skip_remap=True,
    )

    from pprint import pprint
    pprint(results)

if False:
    # Show convo with duplicate votes.
    # Shareable demo: https://gist.github.com/patcon/9c1a39291cd75b23722a5379d7cfc3cc
    report_id = CONVOS["tech-politics-2018"]["report_id"]
    print(f"Loading data from report: https://pol.is/report/{report_id}")

    client = PolisClient(is_strict_moderation=False)
    client.load_data(report_id=report_id, data_source="csv_export")
    client.get_matrix(is_filtered=True)

if False:
    client = PolisClient()
    client.load_data(conversation_id="9xxwa9jpkm")
    # Reproducing this output: https://github.com/compdemocracy/openData/blob/master/london.youth.policing
    # participant-votes.csv matches, but votes.csv is missing entries.
    # BUG: dates for exports seemingly not matching between matrix build and vote export.
    matrix_raw = client.get_matrix(is_filtered=False, cutoff=1658934741418)
    participants_votes_df = client.build_participants_dataframe(matrix_raw)
    print(participants_votes_df)

# This is a sanity-check for equality of a participants-votes.csv dataframe generated from raw votes vs a downloaded export.
if False:
    # Generate the participants-votes dataframe from raw data
    client = PolisClient(is_strict_moderation=True)
    client.load_data(directory_url="https://raw.githubusercontent.com/compdemocracy/openData/refs/heads/master/london.youth.policing/")
    raw_matrix = client.get_matrix(is_filtered=False)
    # Drop statement columns if no votes.
    raw_matrix = raw_matrix.dropna(axis="columns", how="all")
    # Convert all int columns to strings for easier comparison.
    raw_matrix.columns = raw_matrix.columns.astype(str)
    participants_votes_generated = client.build_participants_dataframe(vote_matrix=raw_matrix)
    participants_votes_generated = participants_votes_generated.join(raw_matrix)
    # Remove rows for participants with zero votes.
    non_voting_participant_ids = participants_votes_generated[participants_votes_generated["n_votes"] == 0].index
    participants_votes_generated = participants_votes_generated.drop(index=non_voting_participant_ids)

    # Generate dataframe from downloaded CSV.
    col_mapper = {
        "participant": "participant_id",
        "group-id": "group_id",
        "n-comments": "n_comments",
        "n-votes": "n_votes",
        "n-agree": "n_agree",
        "n-disagree": "n_disagree",
    }
    participants_votes_downloaded = (
        pd
            .read_csv("https://raw.githubusercontent.com/compdemocracy/openData/refs/heads/master/london.youth.policing/participants-votes.csv")
            .rename(columns=col_mapper)
            # Override group_id for now, until we can generate and compare.
            .assign(group_id=None)
            .set_index("participant_id")
            .sort_index()
    )
    is_dataframes_equal = participants_votes_downloaded.equals(participants_votes_generated)
    print(f"downloaded and generated dataframes are equal? {is_dataframes_equal}")
    print(participants_votes_downloaded.shape)
    print(participants_votes_generated.shape)
    print(participants_votes_downloaded.compare(participants_votes_generated))

if False:
    client = PolisClient()
    # client.load_data(report_id=CONVOS["rideshare-toronto"]["report_id"])
    client.load_data(report_id=CONVOS["tech-politics-2018"]["report_id"])
    # client.load_data(conversation_id=CONVOS["anthropic-ccai"]["convo_id"])
    matrix_raw = client.get_matrix(is_filtered=False)
    client.matrix = None # Flush matrix
    matrix_filtered = client.get_matrix(is_filtered=True)
    matrix_filtered_imputed = client.impute_missing_votes()

    presenter = DataPresenter()
    # presenter.generate_vote_heatmap(matrix_raw)
    presenter.generate_vote_heatmap(matrix_filtered)
    # presenter.generate_vote_heatmap(matrix_filtered_imputed)


if False:
    client = PolisClient()
    client.load_data(conversation_id=CONVOS["xid-testing"]["convo_id"])
    xids = [12334552, 12334553, 12334554, "foobar"]
    mappings = client.data_loader.fetch_xid_to_pid_mappings(xids)
    for xid, pid in mappings.items():
        print(f"{pid=} => {xid=}")
