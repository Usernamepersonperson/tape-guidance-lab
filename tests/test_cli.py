"""CLI smoke tests and the figures."""

import re
import struct

import pytest

from agvlab.cli import main


def test_cli_run_exit_code_and_report(capsys):
    assert main(["run", "--route", "straight", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "tracking RMS error" in out and "junction choice       correct" in out


def test_cli_run_reports_failure_exit_code(capsys):
    assert main(["run", "--latency-ms", "400"]) == 1


def test_cli_candump_format(capsys):
    assert main(["candump", "--duration", "0.1", "--iface", "can0"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines and all(re.fullmatch(r"\(\d+\.\d{6}\) can0 [0-9A-F]{3}#[0-9A-F]*", ln) for ln in lines)
    assert {ln.split()[2].split("#")[0] for ln in lines} >= {"195", "295", "395", "715"}


def test_cli_quick_sweep_writes_json(tmp_path):
    out = tmp_path / "r.json"
    assert main(["sweep", "speed", "--quick", "--seeds", "2", "--workers", "1", "--out", str(out)]) == 0
    assert '"speed"' in out.read_text()


@pytest.mark.slow
def test_figure_is_1600x900_png(tmp_path):
    from agvlab.plots import make_figure
    path = tmp_path / "figure.png"
    metrics = make_figure(str(path))
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (1600, 900)
    assert metrics["branch"]["success"] == 1.0


def test_cli_eds_writes_the_object_dictionary(tmp_path):
    out = tmp_path / "x.eds"
    assert main(["eds", "--out", str(out)]) == 0
    assert "[2000sub1]" in out.read_text()


def test_cli_baseline_controller_runs(capsys):
    main(["run", "--controller", "p_only", "--route", "straight"])
    assert "controller p_only" in capsys.readouterr().out


@pytest.mark.slow
def test_post_figure_is_960x700_png(tmp_path):
    from agvlab.plots import make_post_figure
    path = tmp_path / "post.png"
    metrics = make_post_figure(str(path))
    data = path.read_bytes()
    assert struct.unpack(">II", data[16:24]) == (960, 700)
    assert metrics["branch"]["success"] == 1.0
