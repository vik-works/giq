# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""``giq prepare skintokens``: the checkout at its pinned commit, once, and
the environment checked without a card. Commands are recorded, not run."""

import pytest

from giq_rig import prepare


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    """The step with an interpreter in place, its commands recorded."""
    commands = []
    python = tmp_path / "python"
    python.write_text("")
    monkeypatch.setenv("GIQ_SKINTOKENS_PYTHON", str(python))
    monkeypatch.setenv("GIQ_SKINTOKENS_SOURCE", str(tmp_path / "SkinTokens"))
    monkeypatch.setattr("giq.engines.require_binary", lambda engine: str(python))

    def run(command, cwd=None, env=None):
        commands.append((command, cwd, env))
        if command[:2] == ["git", "clone"]:
            (tmp_path / "SkinTokens").mkdir()
            (tmp_path / "SkinTokens" / prepare.MARKER).write_text("")

    monkeypatch.setattr(prepare, "_run", run)
    monkeypatch.setattr(prepare, "_head", lambda source: prepare.SKINTOKENS[1])
    return commands, tmp_path


def test_it_clones_at_the_pinned_commit_and_checks_the_environment(recorded):
    commands, tmp_path = recorded
    assert prepare.main([]) == 0
    url, commit = prepare.SKINTOKENS
    source = str(tmp_path / "SkinTokens")
    clone, pin, check = commands
    assert clone[0] == ["git", "clone", "--quiet", url, source]
    assert pin[0] == ["git", "checkout", "--quiet", commit] and pin[1] == source
    assert check[0][1:] == ["-c", prepare.IMPORTS]
    assert check[2]["CUDA_VISIBLE_DEVICES"] == "", "the check never sees a card"


def test_a_second_run_clones_nothing(recorded):
    commands, _ = recorded
    assert prepare.main([]) == 0
    commands.clear()
    assert prepare.main([]) == 0
    assert len(commands) == 1, "only the environment check runs again"
    assert not any(c[0][0] == "git" for c in commands)


def test_a_checkout_at_another_commit_is_named_and_left_alone(recorded, monkeypatch, capsys):
    commands, tmp_path = recorded
    (tmp_path / "SkinTokens").mkdir()
    (tmp_path / "SkinTokens" / prepare.MARKER).write_text("")
    monkeypatch.setattr(prepare, "_head", lambda source: "0123456789abcdef")
    assert prepare.main([]) == 0
    assert "not the pinned" in capsys.readouterr().out
    assert not any(c[0][0] == "git" for c in commands)


def test_a_folder_in_the_way_is_not_cloned_over(recorded, capsys):
    commands, tmp_path = recorded
    (tmp_path / "SkinTokens").mkdir()
    (tmp_path / "SkinTokens" / "notes.txt").write_text("mine")
    assert prepare.main([]) == 1
    assert "is in the way" in capsys.readouterr().out
    assert commands == []


def test_without_the_interpreter_it_says_how_to_get_one(monkeypatch, capsys):
    def missing(engine):
        raise FileNotFoundError("no interpreter for skintokens")

    monkeypatch.setattr("giq.engines.require_binary", missing)
    assert prepare.main([]) == 1
    assert "cd envs/skintokens && uv sync" in capsys.readouterr().out


def test_giq_prepare_reaches_it():
    from giq import plugins

    assert plugins.engine("skintokens").prepare(["--help"]) == 0
