import pytest

from football_odds.cli.main import build_parser


def test_definitive_command_surface() -> None:
    parser = build_parser()
    for arguments in (
        ["run"],
        ["ingest"],
        ["players", "dataset"],
        ["enrich", "weather"],
        ["market", "build"],
        ["model", "train"],
        ["strategy", "discover"],
    ):
        assert parser.parse_args(arguments).command == arguments[0]
    help_text = parser.format_help()
    for removed in (
        "sport-model",
        "hybrid-model",
        "confirmed-lineup-model",
        "neural-lineup-model",
    ):
        assert removed not in help_text


def test_reserved_commands_have_no_fake_arguments():
    parser = build_parser()
    assert parser.parse_args(["model", "predict"]).action == "predict"
    with pytest.raises(SystemExit):
        parser.parse_args(["model", "predict", "--fixtures", "x.csv"])
