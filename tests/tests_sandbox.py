import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import UnsafeChartCode, execute_chart_code  # noqa: E402


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})


@pytest.fixture
def save_path():
    d = tempfile.mkdtemp(prefix="chart_jail_")
    return os.path.join(d, "chart.png")


# ---------------------------------------------------------------------------
# Escapes that WORKED before hardening
# ---------------------------------------------------------------------------

def test_blocks_subclass_walk_escape(df, save_path):
    """The classic escape: walk from a tuple up to `object`, enumerate every
    loaded subclass, find one holding a live module, and pull the REAL
    __builtins__ (with __import__) out of it. Verified to reach os.getuid()
    as root before the AST guard existed."""
    evil = """
subs = ().__class__.__bases__[0].__subclasses__()
for c in subs:
    if c.__name__ == "catch_warnings":
        os_mod = c()._module.__builtins__["__import__"]("os")
        os_mod.getuid()
"""
    with pytest.raises(UnsafeChartCode):
        execute_chart_code(evil, df, save_path)


def test_blocks_savefig_write_outside_charts_dir(df, save_path):
    """matplotlib is handed in legitimately and already holds filesystem
    access, so removing __import__/open does nothing to stop this. Verified
    to write /tmp/ESCAPED_WRITE.png before the write jail existed."""
    target = os.path.join(tempfile.gettempdir(), "SHOULD_NOT_EXIST_escape.png")
    if os.path.exists(target):
        os.remove(target)

    evil = f'plt.plot(df["a"])\nplt.savefig("{target}")\nplt.close()'
    with pytest.raises(UnsafeChartCode):
        execute_chart_code(evil, df, save_path)
    assert not os.path.exists(target), "write jail failed -- file was created outside charts dir"


def test_blocks_path_traversal_in_savefig(df, save_path):
    """'..' normalizes away on resolve(), so this must be caught too."""
    evil = 'plt.plot(df["a"])\nplt.savefig(save_path + "/../../../../tmp/traversal.png")\nplt.close()'
    with pytest.raises(UnsafeChartCode):
        execute_chart_code(evil, df, save_path)


# ---------------------------------------------------------------------------
# Static guard: things that must never reach exec()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "snippet,label",
    [
        ("import os", "import statement"),
        ("from os import system", "from-import"),
        ('x = ().__class__', "__class__ access"),
        ('x = df.__dict__', "__dict__ access"),
        ('f = open("/etc/passwd")', "open()"),
        ('eval("1+1")', "eval()"),
        ('exec("x=1")', "exec()"),
        ('getattr(df, "to_csv")("/tmp/leak.csv")', "getattr()"),
        ('g = globals()', "globals()"),
        ('x = getattr(df, "__cl" + "ass__")', "split-string dunder via getattr"),
        ('def helper():\n    pass', "function definition"),
        ('f = lambda: 1', "lambda"),
        ('class X:\n    pass', "class definition"),
    ],
)
def test_static_guard_rejects(df, save_path, snippet, label):
    with pytest.raises(UnsafeChartCode):
        execute_chart_code(snippet, df, save_path)


def test_rejects_unparseable_code(df, save_path):
    with pytest.raises(UnsafeChartCode):
        execute_chart_code("plt.plot(df['a'']", df, save_path)


# ---------------------------------------------------------------------------
# The guarded __import__ (needed for numpy's lazy imports) must stay narrow
# ---------------------------------------------------------------------------

def test_guarded_import_allows_plotting_stack():
    """numpy defers part of itself -- .mean() imports numpy._core._methods
    through the sandbox's own __builtins__. This must resolve."""
    from src.graph import _guarded_import

    assert _guarded_import("numpy") is not None
    assert _guarded_import("matplotlib.cm") is not None


@pytest.mark.parametrize("mod", ["os", "sys", "subprocess", "importlib", "builtins", "socket", "shutil"])
def test_guarded_import_blocks_everything_else(mod):
    from src.graph import _guarded_import

    with pytest.raises(ImportError):
        _guarded_import(mod)


def test_numpy_mean_works_in_chart_code(df, save_path):
    """Regression test for the KeyError: '__import__' bug -- this exact
    snippet failed before the guarded import was added."""
    code = """
arr = np.array(df["a"])
plt.axhline(y=float(arr.mean()))
plt.plot(df["a"])
plt.tight_layout()
plt.savefig(save_path)
plt.close()
"""
    execute_chart_code(code, df, save_path)
    assert os.path.exists(save_path)


# ---------------------------------------------------------------------------
# The guard must not break legitimate chart code
# ---------------------------------------------------------------------------

def test_allows_normal_matplotlib(df, save_path):
    code = """
plt.figure(figsize=(6, 4))
plt.plot(df["a"], df["b"], marker="o")
plt.xlabel("a")
plt.ylabel("b")
plt.title("b vs a")
plt.tight_layout()
plt.savefig(save_path)
plt.close()
"""
    execute_chart_code(code, df, save_path)
    assert os.path.exists(save_path), "legitimate chart failed to save"


def test_allows_seaborn_and_pandas_idioms(df, save_path):
    code = """
means = df.groupby("a")["b"].mean().reset_index()
sns.barplot(data=means, x="a", y="b")
plt.tight_layout()
plt.savefig(save_path)
plt.close()
"""
    execute_chart_code(code, df, save_path)
    assert os.path.exists(save_path)


def test_allows_numpy_and_control_flow(df, save_path):
    """Loops, comprehensions and conditionals are all legitimate in chart
    code -- the guard must not be so blunt that it rejects them."""
    code = """
vals = [x * 2 for x in df["a"] if x > 1]
arr = np.array(vals)
for i in range(2):
    plt.axhline(y=float(arr.mean()) + i, linestyle="--")
plt.bar(range(len(vals)), vals)
plt.tight_layout()
plt.savefig(save_path)
plt.close()
"""
    execute_chart_code(code, df, save_path)
    assert os.path.exists(save_path)


def test_timeout_stops_infinite_loop(df, save_path):
    """A runaway loop must be killed rather than hanging the whole pipeline."""
    import signal as _signal

    if not hasattr(_signal, "SIGALRM"):
        pytest.skip("SIGALRM unavailable on this platform")
    code = "while True:\n    pass"
    with pytest.raises(Exception) as exc:
        execute_chart_code(code, df, save_path, timeout_seconds=2)
    assert "too long" in str(exc.value).lower() or "timeout" in type(exc.value).__name__.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))