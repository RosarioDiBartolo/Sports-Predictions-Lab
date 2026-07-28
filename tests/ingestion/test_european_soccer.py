import sqlite3

from football_odds.ingestion.providers.european_soccer import (
    import_european_soccer_database,
)


def _source(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE Country (id INTEGER, name TEXT);
        CREATE TABLE Team (
            id INTEGER, team_api_id INTEGER, team_long_name TEXT
        );
        CREATE TABLE Player (player_api_id INTEGER, player_name TEXT);
        """
    )
    player_columns = ", ".join(
        f"{side}_player_{number} INTEGER, "
        f"{side}_player_X{number} REAL, "
        f"{side}_player_Y{number} REAL"
        for side in ("home", "away")
        for number in range(1, 12)
    )
    connection.execute(
        f"""
        CREATE TABLE Match (
            match_api_id INTEGER, country_id INTEGER, season TEXT, date TEXT,
            home_team_api_id INTEGER, away_team_api_id INTEGER,
            home_team_goal INTEGER, away_team_goal INTEGER,
            {player_columns}
        )
        """
    )
    connection.execute("INSERT INTO Country VALUES (1, 'England')")
    connection.executemany(
        "INSERT INTO Team VALUES (?, ?, ?)",
        [(1, 10, "Home FC"), (2, 20, "Away FC")],
    )
    connection.executemany(
        "INSERT INTO Player VALUES (?, ?)",
        [(number, f"Player {number}") for number in range(1, 23)],
    )
    columns = [
        "match_api_id",
        "country_id",
        "season",
        "date",
        "home_team_api_id",
        "away_team_api_id",
        "home_team_goal",
        "away_team_goal",
    ]
    values = [900, 1, "2015/2016", "2016-01-02 15:00:00", 10, 20, 2, 1]
    for side, offset in (("home", 0), ("away", 11)):
        for number in range(1, 12):
            columns.extend(
                [
                    f"{side}_player_{number}",
                    f"{side}_player_X{number}",
                    f"{side}_player_Y{number}",
                ]
            )
            y = 1 if number == 1 else 3 if number <= 5 else 7 if number <= 9 else 10
            values.extend([offset + number, number, y])
    placeholders = ",".join("?" for _ in values)
    connection.execute(
        f"INSERT INTO Match ({','.join(columns)}) VALUES ({placeholders})",
        values,
    )
    connection.commit()
    connection.close()


def test_imports_complete_elevens_with_derived_roles_and_resumes(tmp_path):
    source = tmp_path / "source.sqlite"
    database = tmp_path / "canonical.sqlite"
    _source(source)

    result = import_european_soccer_database(
        tmp_path,
        source_path=source,
        database_path=database,
        seasons=("2015/2016",),
    )
    assert result.matches_imported == 1
    assert result.player_observations == 22

    connection = sqlite3.connect(database)
    assert dict(
        connection.execute(
            "SELECT position, COUNT(*) FROM lineup_players GROUP BY position"
        ).fetchall()
    ) == {"D": 8, "F": 4, "G": 2, "M": 8}
    connection.close()

    resumed = import_european_soccer_database(
        tmp_path,
        source_path=source,
        database_path=database,
        seasons=("2015/2016",),
    )
    assert resumed.matches_imported == 0
    assert resumed.already_imported == 1
