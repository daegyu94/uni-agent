from typing import Any

import pytest

from uni_agent.sandbox import ExecResult
from uni_agent.tools.shell import TmuxShell


class RecordingBackend:
    def __init__(self) -> None:
        self.exec_calls: list[list[str]] = []
        self.writes: list[tuple[str, bytes | str]] = []

    async def exec(self, argv: list[str], **kwargs: Any) -> ExecResult:
        self.exec_calls.append(list(argv))
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.writes.append((path, content))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_start_command_keeps_large_payload_out_of_tmux_argv():
    backend = RecordingBackend()
    channel = TmuxShell(backend, session_id="test-session")  # type: ignore[arg-type]
    command = "printf 'large payload\\n'\n" * 10_000

    command_id = await channel.start_command(command)

    command_path = "/tmp/uni-agent-shell/test-session/cmd_1.input"
    assert command_id == 1
    assert backend.writes == [(command_path, command)]
    assert len(backend.exec_calls) == 1

    send_keys_argv = backend.exec_calls[0]
    injected_line = send_keys_argv[-2]
    assert send_keys_argv[-1] == "Enter"
    assert f'eval "$(cat {command_path})"' in injected_line
    assert "large payload" not in injected_line
    assert len(injected_line) < 1_000


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_interrupt_returns_when_ctrl_c_finishes_command(monkeypatch: pytest.MonkeyPatch):
    backend = RecordingBackend()
    channel = TmuxShell(backend, session_id="test-session")  # type: ignore[arg-type]

    async def sleep(_seconds: float) -> None:
        return None

    async def poll(_command_id: int) -> int:
        return 130

    monkeypatch.setattr("uni_agent.tools.shell.asyncio.sleep", sleep)
    monkeypatch.setattr(channel, "poll", poll)

    code = await channel.interrupt(1)

    assert code == 130
    send_keys_calls = [call for call in backend.exec_calls if "send-keys" in call]
    assert [call[-1] for call in send_keys_calls] == ["C-c"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_interrupt_suspends_and_kills_job_when_ctrl_c_fails(monkeypatch: pytest.MonkeyPatch):
    backend = RecordingBackend()
    channel = TmuxShell(backend, session_id="test-session")  # type: ignore[arg-type]
    poll_results = iter([None, 124])

    async def sleep(_seconds: float) -> None:
        return None

    async def poll(_command_id: int) -> int | None:
        return next(poll_results)

    monkeypatch.setattr("uni_agent.tools.shell.asyncio.sleep", sleep)
    monkeypatch.setattr(channel, "poll", poll)

    code = await channel.interrupt(1)

    assert code == 124
    send_keys_calls = [call for call in backend.exec_calls if "send-keys" in call]
    assert send_keys_calls[0][-1] == "C-c"
    assert send_keys_calls[1][-1] == "C-z"
    assert send_keys_calls[2][-1] == "Enter"
    fallback_line = send_keys_calls[2][-2]
    assert "kill -KILL %+" in fallback_line


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_run_marks_timeout_even_when_interrupt_reports_an_exit_code(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = RecordingBackend()
    channel = TmuxShell(backend, session_id="test-session")  # type: ignore[arg-type]

    async def start_command(_command: str) -> int:
        return 1

    async def poll(_command_id: int) -> None:
        return None

    async def interrupt(_command_id: int) -> int:
        return 130

    monkeypatch.setattr(channel, "start_command", start_command)
    monkeypatch.setattr(channel, "poll", poll)
    monkeypatch.setattr(channel, "interrupt", interrupt)

    result = await channel.run("sleep forever", timeout=0.0)

    assert result.timed_out is True
    assert result.exit_code == 130


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"o": "out\n", "e": "warn: é\n"}, ("out\n", "warn: é\n")),
        ({"e": "only err"}, ("", "only err")),
        ({"o": "only out"}, ("only out", "")),
        ({}, ("", "")),
    ],
)
async def test_read_outputs_returns_both_capture_files_in_one_exec(tmp_path, files, expected):
    from uni_agent.sandbox.local import LocalSandbox

    class CountingLocal(LocalSandbox):
        calls = 0

        async def exec(self, argv, **kwargs):
            CountingLocal.calls += 1
            return await super().exec(argv, **kwargs)

    for name, text in files.items():
        (tmp_path / name).write_text(text)
    channel = TmuxShell(CountingLocal(), session_id="test-session")

    assert await channel._read_outputs(str(tmp_path / "o"), str(tmp_path / "e")) == expected
    assert CountingLocal.calls == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_read_outputs_treats_failed_exec_as_empty():
    class FailingBackend(RecordingBackend):
        async def exec(self, argv: list[str], **kwargs: Any) -> ExecResult:
            return ExecResult(exit_code=127, stdout="", stderr="backend error")

    channel = TmuxShell(FailingBackend(), session_id="test-session")  # type: ignore[arg-type]

    assert await channel._read_outputs("/x.out", "/x.err") == ("", "")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_run_collects_outputs_with_a_single_exec(monkeypatch: pytest.MonkeyPatch):
    class OutputBackend(RecordingBackend):
        async def exec(self, argv: list[str], **kwargs: Any) -> ExecResult:
            await super().exec(argv, **kwargs)
            return ExecResult(exit_code=0, stdout="captured out", stderr="captured err")

    backend = OutputBackend()
    channel = TmuxShell(backend, session_id="test-session")  # type: ignore[arg-type]

    async def poll(_command_id: int) -> int:
        return 0

    monkeypatch.setattr(channel, "poll", poll)

    result = await channel.run("true")

    assert (result.stdout, result.stderr) == ("captured out", "captured err")
    send_keys, read = backend.exec_calls
    assert send_keys[3] == "send-keys"
    assert read[:2] == ["sh", "-c"]
    assert read[-2:] == ["/tmp/uni-agent-shell/test-session/cmd_1.out", "/tmp/uni-agent-shell/test-session/cmd_1.err"]
