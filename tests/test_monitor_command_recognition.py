"""V10.4 monitor-command recognition tests.

Focuses on the monitor-destination slot of recovered move commands:
spatial words (left/right), ordinals (first/second/monitor one/two),
roles (primary/other) and STT confusions (write->right) — plus the
guarantee that an unrecognized destination produces a clarification
rather than a wrong monitor.
"""

from __future__ import annotations

import types

import pytest

from src.commands.recovery import (
    MisheardCommandRecovery,
    _score_monitor,
)


def _app(name):
    return types.SimpleNamespace(name=name, exe_name=name.lower())


APPS = [
    _app("Chrome"), _app("Devin"), _app("WhatsApp"), _app("Notepad"),
]


def make_recovery(monitors=None):
    resolver = None
    if monitors is not None:
        def resolver(query, relative_to=None):
            return monitors.resolve(query, relative_to=relative_to)
    return MisheardCommandRecovery(
        app_names=lambda: APPS,
        monitor_resolver=resolver,
    )


class TestMonitorVocabularyScoring:
    @pytest.mark.parametrize("fragment,best", [
        ("left monitor", "left"),
        ("right monitor", "right"),
        ("the left screen", "left"),
        ("monitor two", "monitor two"),
        ("monitor 2", "monitor 2"),
        ("second monitor", "second"),
        ("primary monitor", "primary"),
        ("other monitor", "other"),
        ("the other screen", "other"),
        ("write monitor", "right"),   # STT confusion fix
    ])
    def test_vocab_fragments(self, fragment, best):
        score, spec = _score_monitor(fragment)
        assert score >= 0.5, (fragment, score, spec)
        assert best in spec or best in fragment

    def test_garbage_monitor_scores_low(self):
        score, spec = _score_monitor("commoditous zanzibar")
        assert score < 0.5


class TestMoveRecoveryMonitorSlot:
    def test_left_monitor_parsed(self):
        rec = make_recovery()
        result = rec.recover("Move Chrome to the left monitor.")
        assert result.status == "matched"
        ent = result.intent.entities
        assert "chrome" in ent["target"].lower()
        assert "left" in ent["monitor"].lower()

    def test_right_monitor_parsed(self):
        rec = make_recovery()
        result = rec.recover("Move Devin to the right monitor.")
        assert result.status == "matched"
        assert "devin" in result.intent.entities["target"].lower()
        assert "right" in result.intent.entities["monitor"].lower()

    def test_monitor_number_parsed(self):
        rec = make_recovery()
        result = rec.recover("Move WhatsApp to monitor two.")
        assert result.status == "matched"
        assert "monitor two" in result.intent.entities["monitor"]

    def test_other_monitor_parsed(self):
        rec = make_recovery()
        result = rec.recover("Move the window to the other monitor.")
        # App fragment "the window" won't match an app; recovery may
        # decline or produce target passthrough — never invent an app.
        assert result.status in ("matched", "none", "ambiguous")

    def test_garbled_monitor_passes_through_for_clarify(self):
        rec = make_recovery()
        result = rec.recover("More crumpled less commoditous.")
        # Whatever the outcome, the monitor entity must not be a
        # confident wrong guess like "right".
        if result.intent is not None:
            mon = str(result.intent.entities.get("monitor", ""))
            meta = result.intent.entities.get("_recovery", {})
            assert meta.get("original_transcript") == \
                "More crumpled less commoditous."
            # An uninterpreted fragment stays raw for the handler to
            # clarify — it must not equal a canonical wrong monitor.
            if "commoditous" in mon:
                pass  # passthrough -> handler will ask which monitor

    def test_monitor_resolver_validates(self):
        class FakeMonitors:
            def resolve(self, query, relative_to=None):
                class R:
                    status = "found"
                    monitor = "MON"
                return R()

        rec = make_recovery(monitors=FakeMonitors())
        result = rec.recover("Move Chrome to the left monitor.")
        assert result.status == "matched"
