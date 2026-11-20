"""
Static check of the code in README.md, docs/ and examples/ against the API.

Every ``python`` block and example script is parsed (not run). Calls to
psplines functions and to the ``PSpline`` methods ``predict``, ``derivative``
and ``fit`` may only use keyword arguments the real signature has, and no
code may call the Bayesian API removed in 0.3.0. This catches calls such as
``predict(..., hdi_prob=0.95)`` or ``spline.bayes_fit()``.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path

import pytest

import psplines
from psplines import PSpline, WhittakerSmoother, basis, utils_math

ROOT = Path(__file__).resolve().parent.parent
_BLOCK = re.compile(r"^([ \t]*)```(?:python|py)[ \t]*\n(.*?)^[ \t]*```", re.M | re.S)


def _sources() -> list[tuple[str, int, str]]:
    """(file, first line, code) for every python block and example script."""
    out = []
    md = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for path in md:
        text = path.read_text(encoding="utf-8")
        for m in _BLOCK.finditer(text):
            line = text.count("\n", 0, m.start(2)) + 1
            code = textwrap.dedent(m.group(2))
            out.append((str(path.relative_to(ROOT)), line, code))
    for path in sorted((ROOT / "examples").glob("*.py")):
        out.append((str(path.relative_to(ROOT)), 1, path.read_text(encoding="utf-8")))
    return out


def _params(*funcs) -> set[str] | None:
    """Union of keyword names, or None if any accepts **kwargs."""
    names: set[str] = set()
    for f in funcs:
        for p in inspect.signature(f).parameters.values():
            if p.kind is p.VAR_KEYWORD:
                return None
            if p.kind is not p.POSITIONAL_ONLY:
                names.add(p.name)
    return names


METHODS = {
    "predict": _params(PSpline.predict, WhittakerSmoother.predict),
    "derivative": _params(PSpline.derivative),
    "fit": _params(PSpline.fit, WhittakerSmoother.fit),
}
# Removed in 0.3.0 together with the Bayesian fit.
REMOVED_METHODS = {
    "bayes_fit",
    "posterior_draws",
    "plot_lam_trace",
    "plot_alpha_trace",
    "plot_posterior",
}
FUNCTIONS = {
    name: _params(obj)
    for name, obj in {
        **{n: getattr(psplines, n) for n in psplines.__all__},
        "b_spline_derivative_basis": basis.b_spline_derivative_basis,
        "effective_df": utils_math.effective_df,
    }.items()
    if callable(obj) and not (inspect.isclass(obj) and issubclass(obj, Exception))
}


def _problems(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # illustrative fragment, not runnable code
    local = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in REMOVED_METHODS:
                found.append(f"line +{node.lineno - 1}: {f.attr}() was removed")
                continue
            for kw in node.keywords:
                if (
                    kw.arg == "se_method"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "bayes"
                ):
                    found.append(f"line +{node.lineno - 1}: se_method='bayes'")
            if isinstance(f, ast.Attribute) and f.attr in METHODS:
                name, allowed = f.attr, METHODS[f.attr]
            elif isinstance(f, ast.Name) and f.id in FUNCTIONS and f.id not in local:
                name, allowed = f.id, FUNCTIONS[f.id]
            else:
                continue
            if allowed is None:
                continue
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in allowed:
                    found.append(f"line +{node.lineno - 1}: {name}(..., {kw.arg}=...)")
    return found


SOURCES = _sources()


def test_sources_found():
    files = {f for f, _, _ in SOURCES}
    assert "README.md" in files
    assert any(f.startswith("examples/") for f in files)
    assert any(f.startswith("docs/tutorials/") for f in files)


@pytest.mark.parametrize(
    ("where", "code"),
    [(f"{f}:{line}", code) for f, line, code in SOURCES],
    ids=[f"{f}:{line}" for f, line, _ in SOURCES],
)
def test_doc_code_matches_api(where, code):
    assert _problems(code) == [], where


def test_no_removed_install_extras():
    """The bayes and nutpie extras were removed in 0.3.0 (the README
    changelog may still name them)."""
    pattern = re.compile(r"psplines\[(?:[^\]]*,)?\s*(bayes|nutpie)\b")
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        if path.name == "README.md":
            text = text.split("## Changelog")[0]
        assert not pattern.search(text), path.relative_to(ROOT)
