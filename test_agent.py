from datetime import date
from agent import find_next_match


def test_find_next_match():
    text = '''2026 Schedule\nUpcoming\nHOME\nSaturday\nOctober\n24\n7:30 PM\nOrange County SC\nat Toyota Field\nTICKETS'''
    result = find_next_match(text, today=date(2026, 10, 17))
    assert result["opponent"] == "Orange County SC"
    assert result["date"] == "2026-10-24"
    assert result["venue"] == "at Toyota Field"
