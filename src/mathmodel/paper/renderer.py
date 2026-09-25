"""MathModel AI — Phase 6E: LaTeX renderer.

PaperIR → paper.tex. If a LaTeX compiler exists, compile to PDF and
record a CompilationRecord. Never fake a PDF.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from mathmodel.paper import PaperIR, PaperSection, BlockType


@dataclass
class CompilationRecord:
    """Record of a LaTeX compilation attempt."""
    compilation_id: str = field(default_factory=lambda: f"COMP-{uuid4().hex[:8]}")
    command: str = ""
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    warnings: list[str] = field(default_factory=list)
    pdf_path: str = ""
    pdf_hash: str = ""
    success: bool = False
    compiler_available: bool = False
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


_INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters, preserving inline math.

    Model-written prose contains inline math such as ``$\\Delta t$``. Escaping
    the dollar signs turns real notation into the literal characters
    ``$\\textbackslash{}Delta t$`` on the page, so ``$...$`` spans are passed
    through untouched. The injection guard still scans the whole document, so
    nothing dangerous can hide inside a math span.
    """
    if "$" not in text:
        return _escape_latex_chars(text)

    parts: list[str] = []
    cursor = 0
    for match in _INLINE_MATH_RE.finditer(text):
        parts.append(_escape_latex_chars(text[cursor:match.start()]))
        parts.append("$" + match.group(1) + "$")
        cursor = match.end()
    parts.append(_escape_latex_chars(text[cursor:]))
    return "".join(parts)


def _escape_latex_chars(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


# Dangerous LaTeX primitives that must never pass through from content.
# Each entry is matched as a COMPLETE control sequence: LaTeX command names are
# runs of letters, so "\include" must not fire on the legitimate
# "\includegraphics", and "\input" must not fire on "\inputencoding".
FORBIDDEN_TEX_PATTERNS = [
    r"\input",
    r"\include",
    r"\write18",
    r"\write",
    r"\openin",
    r"\openout",
    r"\read",
    r"\catcode",
    r"\immediate\write",
    r"\pdfprimitive",
    r"\special{",
]

_FORBIDDEN_TEX_RE = re.compile(
    "|".join(
        # Only a pattern ending in a letter is a bare command name, and only
        # those need the "not followed by another letter" guard. A pattern that
        # already carries its own delimiter ("\special{") is complete as written.
        re.escape(pattern) + (r"(?![A-Za-z])" if pattern[-1].isalpha() else "")
        for pattern in FORBIDDEN_TEX_PATTERNS
    )
)


def _check_latex_injection(tex: str) -> list[str]:
    """Detect dangerous LaTeX commands. Returns list of issues."""
    issues = []
    for match in _FORBIDDEN_TEX_RE.finditer(tex):
        issues.append(f"Dangerous LaTeX command detected: {match.group(0)}")
    return issues


class LaTeXInjectionError(Exception):
    """Raised when paper content attempts LaTeX injection."""
    pass


class LaTeXRenderer:
    """Renders PaperIR into LaTeX and optionally compiles to PDF."""

    def __init__(self):
        self._section_counter: dict[str, int] = {}
        self._equation_number = 0
        self._figure_number = 0
        self._table_number = 0

    def render(self, paper: PaperIR) -> str:
        """Render PaperIR to LaTeX source."""
        lines = []
        lines.append(r"\documentclass{article}")
        lines.append(r"\usepackage[utf8]{inputenc}")
        lines.append(r"\usepackage{amsmath,amssymb}")
        lines.append(r"\usepackage{graphicx}")
        lines.append(r"\usepackage{booktabs}")
        lines.append(r"\usepackage[colorlinks=true]{hyperref}")
        lines.append("")
        lines.append(r"\begin{document}")
        lines.append("")
        lines.append(r"\title{" + _latex_escape(paper.title) + "}")
        lines.append(r"\maketitle")
        lines.append("")
        if paper.abstract:
            lines.append(r"\begin{abstract}")
            lines.append(_latex_escape(paper.abstract))
            lines.append(r"\end{abstract}")
            lines.append("")
        if paper.keywords:
            lines.append(r"\noindent\textbf{Keywords:} " + _latex_escape(", ".join(paper.keywords)))
            lines.append("")

        for section in paper.sections:
            lines.extend(self._render_section(section))

        if paper.references:
            lines.append(r"\begin{thebibliography}{99}")
            for i, ref_id in enumerate(paper.references, 1):
                lines.append(rf"\bibitem{{ref{i}}} {_latex_escape(ref_id)}")
            lines.append(r"\end{thebibliography}")

        for appendix in paper.appendices:
            lines.append(r"\appendix")
            lines.extend(self._render_section(appendix))

        lines.append(r"\end{document}")
        tex = "\n".join(lines)

        # Security: block dangerous LaTeX commands in the final document
        issues = _check_latex_injection(tex)
        if issues:
            raise LaTeXInjectionError("; ".join(issues))

        return tex

    def _render_section(self, section: PaperSection) -> list[str]:
        lines = []
        lines.append(rf"\section{{{_latex_escape(section.title)}}}")
        lines.append("")
        for block in section.content_blocks:
            lines.extend(self._render_block(block))
        lines.append("")
        return lines

    def _render_block(self, block) -> list[str]:
        if block.block_type == BlockType.PARAGRAPH:
            return [_latex_escape(block.text), ""]
        if block.block_type == BlockType.EQUATION:
            self._equation_number += 1
            return [
                r"\begin{equation}",
                block.text,  # Equation LaTeX — NOT escaped (already LaTeX)
                r"\end{equation}",
                "",
            ]
        if block.block_type == BlockType.FIGURE:
            self._figure_number += 1
            return [
                r"\begin{figure}[htbp]",
                r"\centering",
                r"\fbox{Figure placeholder}",
                rf"\caption{{{_latex_escape(block.text)}}}",
                r"\end{figure}",
                "",
            ]
        if block.block_type == BlockType.TABLE:
            self._table_number += 1
            return [
                r"\begin{table}[htbp]",
                r"\centering",
                rf"\caption{{{_latex_escape(block.text)}}}",
                r"\end{table}",
                "",
            ]
        if block.block_type == BlockType.LIST:
            lines = [r"\begin{itemize}"]
            for item in block.items:
                lines.append(rf"\item {_latex_escape(item)}")
            lines.append(r"\end{itemize}")
            lines.append("")
            return lines
        if block.block_type == BlockType.NOTE:
            return [r"\textbf{Note:} " + _latex_escape(block.text), ""]
        return [_latex_escape(block.text), ""]

    def compile(
        self,
        tex_source: str,
        output_dir: Optional[str] = None,
    ) -> CompilationRecord:
        """Compile LaTeX to PDF if a compiler is available."""
        record = CompilationRecord()

        compiler = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("latexmk")
        if not compiler:
            record.compiler_available = False
            record.metadata["reason"] = "No LaTeX compiler found on PATH"
            return record

        record.compiler_available = True
        work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="latex_"))
        work_dir.mkdir(parents=True, exist_ok=True)

        tex_path = work_dir / "paper.tex"
        tex_path.write_text(tex_source, encoding="utf-8")

        record.started_at = datetime.now(timezone.utc)
        record.command = f"{compiler} paper.tex"

        try:
            result = subprocess.run(
                [compiler, "-interaction=nonstopmode", "paper.tex"],
                capture_output=True, text=True, timeout=120,
                cwd=str(work_dir),
                encoding="utf-8", errors="replace",
            )
            record.exit_code = result.returncode
            record.stdout = (result.stdout or "")[-20_000:]
            record.stderr = (result.stderr or "")[-20_000:]
            record.success = result.returncode == 0

            pdf_path = work_dir / "paper.pdf"
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                record.pdf_path = str(pdf_path)
                record.pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
            else:
                record.success = False
                record.metadata["reason"] = "No PDF produced"

            # Extract warnings
            for line in result.stdout.splitlines():
                if "Warning" in line:
                    record.warnings.append(line.strip()[:200])

        except subprocess.TimeoutExpired:
            record.exit_code = -1
            record.success = False
            record.metadata["reason"] = "Compilation timed out"
        except Exception as e:
            record.exit_code = -1
            record.success = False
            record.metadata["reason"] = str(e)
        finally:
            record.finished_at = datetime.now(timezone.utc)

        return record