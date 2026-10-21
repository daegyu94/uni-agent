"""Path validation for ``str_replace_editor`` against a real local sandbox."""

from pathlib import Path

import pytest

from uni_agent.sandbox.local import LocalSandbox
from uni_agent.tools.edit_file import EditFileTool


class CountingSandbox(LocalSandbox):
    def __init__(self) -> None:
        super().__init__()
        self.exec_calls: list[list[str]] = []

    async def exec(self, argv, **kwargs):
        self.exec_calls.append(list(argv))
        return await super().exec(argv, **kwargs)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("a = 1\nb = 2\n")
    (tmp_path / "-dash file.py").write_text("x = 1\n")
    (tmp_path / "link").symlink_to(tmp_path / "pkg")
    return tmp_path


async def _run(sandbox: CountingSandbox, **args) -> str:
    return (await EditFileTool(sandbox).run(args)).text


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_view_file_probes_path_once(tree: Path):
    sandbox = CountingSandbox()

    text = await _run(sandbox, command="view", path=str(tree / "pkg" / "mod.py"))

    assert "a = 1" in text
    assert len(sandbox.exec_calls) == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["pkg", "link"])
async def test_view_directory_and_symlinked_directory_list_entries(tree: Path, name: str):
    sandbox = CountingSandbox()

    text = await _run(sandbox, command="view", path=str(tree / name))

    assert "files and directories up to 2 levels deep" in text
    assert [argv[0] for argv in sandbox.exec_calls] == ["sh", "find"]
    assert "not allowed" in await _run(sandbox, command="view", path=str(tree / name), view_range=[1, 2])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_str_replace_probes_path_once(tree: Path):
    sandbox = CountingSandbox()
    path = tree / "-dash file.py"

    text = await _run(sandbox, command="str_replace", path=str(path), old_str="x = 1", new_str="x = 2")

    assert "has been edited" in text
    assert path.read_text() == "x = 2\n"
    assert len(sandbox.exec_calls) == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_path_validation_messages(tree: Path):
    sandbox = CountingSandbox()
    missing, directory, existing = tree / "missing.py", tree / "pkg", tree / "pkg" / "mod.py"

    assert "does not exist" in await _run(sandbox, command="view", path=str(missing))
    assert "File already exists" in await _run(sandbox, command="create", path=str(existing), file_text="")
    assert "File already exists" in await _run(sandbox, command="create", path=str(directory), file_text="")
    assert "is a directory" in await _run(sandbox, command="str_replace", path=str(directory), old_str="a")
    assert "ends with `/`" in await _run(sandbox, command="create", path=f"{tree}/new/", file_text="")
    assert "parent directory" in await _run(sandbox, command="create", path=str(tree / "no" / "f.py"), file_text="")
    assert "created successfully" in await _run(sandbox, command="create", path=str(missing), file_text="y = 1\n")
    assert missing.read_text() == "y = 1\n"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_dangling_symlink_reads_as_missing(tree: Path):
    sandbox = CountingSandbox()
    (tree / "dangling").symlink_to(tree / "gone.py")

    assert "does not exist" in await _run(sandbox, command="view", path=str(tree / "dangling"))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_failed_probe_exec_reads_as_missing(tree: Path):
    class FailingSandbox(CountingSandbox):
        async def _exec(self, argv, **kwargs):
            raise OSError("backend unavailable")

    assert "does not exist" in await _run(FailingSandbox(), command="view", path=str(tree / "pkg" / "mod.py"))
