"""MathModel AI — CUMCM Autopilot: evidence package, figures, paper, delivery.

The paper is assembled from a structured, already-verified context package —
never from raw chat history. Figures and tables are rendered from the real
artifacts produced by the verified execution, and every number in them keeps
a source id that can be traced back to that execution.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from mathmodel.autopilot.verify import read_output_table
from mathmodel.documents import (
    FigureRecord,
    FigureRegistry,
    FigureType,
    TableCell,
    TableRecord,
    TableRegistry,
    TableType,
)
from mathmodel.paper import BlockType, ContentBlock, PaperIR, PaperSection
from mathmodel.routing.profile import TaskProfile, TaskType
from mathmodel.routing.router import ModelRouter

PAPER_COMPILE_IMAGE = "mathmodel-ai-paper:phase6"

DATA_STATUS_FORMAL = "formal"
DATA_STATUS_DEMO = "demo"
DATA_STATUS_BLOCKED = "blocked"


# ═══════════════════════════════════════════════════════════════
# Paper context package
# ═══════════════════════════════════════════════════════════════

class PaperContextPackage(BaseModel):
    """Everything the paper writer is allowed to use. Nothing else."""

    problem_title: str = ""
    problem_text: str = ""
    subproblems: list[dict[str, Any]] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    ambiguities: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    model_summary: dict[str, Any] = Field(default_factory=dict)
    equations: list[dict[str, Any]] = Field(default_factory=list)
    solver_approach: str = ""
    verified_statistics: dict[str, Any] = Field(default_factory=dict)
    verification_summary: dict[str, Any] = Field(default_factory=dict)
    output_files: list[str] = Field(default_factory=list)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    data_status: str = DATA_STATUS_FORMAL

    def available_ids(self) -> dict[str, list[str]]:
        return {
            "figure_ids": [f["figure_id"] for f in self.figures],
            "table_ids": [t["table_id"] for t in self.tables],
            "equation_ids": [e.get("equation_id", "") for e in self.equations],
            "reference_ids": [r.get("reference_id", "") for r in self.references],
        }

    def to_prompt(self) -> str:
        payload = self.model_dump()
        payload["problem_text"] = self.problem_text[:8000]
        return json.dumps(payload, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════

class FigureSpec(BaseModel):
    """A figure the writer wants, bound to real data."""

    title: str
    caption: str = ""
    chart: str = Field(default="bar", description="bar | line | pie | scatter | histogram")
    figure_type: str = Field(default="comparison", description="FigureType enum value")
    source: str = Field(
        default="statistics",
        description="statistics | output_file",
    )
    statistics_keys: list[str] = Field(default_factory=list)
    output_file: str = ""
    output_columns: list[str] = Field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    supported_claim: str = ""


class FigureSpecList(BaseModel):
    figures: list[FigureSpec] = Field(default_factory=list)


class FigureBuilder:
    """Renders figures from verified results, never from prose."""

    def __init__(self, router: Optional[ModelRouter] = None):
        self._router = router

    async def propose(
        self,
        statistics: dict[str, Any],
        output_summaries: dict[str, dict[str, Any]],
        subproblems: list[dict[str, Any]],
    ) -> list[FigureSpec]:
        if self._router is None:
            return []
        prompt = "\n".join([
            "Propose 2-5 figures for a CUMCM competition paper.",
            "",
            "## VERIFIED STATISTICS (all real, computed by executed code)",
            json.dumps(statistics, ensure_ascii=False, indent=2),
            "",
            "## RESULT FILES",
            json.dumps(output_summaries, ensure_ascii=False, indent=2),
            "",
            "## SUBPROBLEMS",
            json.dumps(subproblems, ensure_ascii=False, indent=2),
            "",
            "Rules:",
            "- Every figure MUST be bound to real data that exists above: either "
            "`statistics` with exact `statistics_keys`, or `output_file` with exact "
            "`output_columns`.",
            "- Never propose a figure whose data is not listed above.",
            "- `chart` is one of: bar, line, pie, scatter, histogram.",
            "- `figure_type` is one of: comparison, trend, solution, histogram, "
            "scatter, sensitivity, robustness_distribution, network, flow.",
            "- Write titles and captions in Chinese.",
        ])
        profile = TaskProfile.for_task_type(TaskType.VISUALIZATION)
        try:
            result = await self._router.route_structured_generate(
                profile=profile,
                prompt=prompt,
                output_schema=FigureSpecList,
                system_prompt=(
                    "You design publication-quality figures for mathematical "
                    "modeling papers. You only propose figures whose data exists."
                ),
            )
        except Exception:
            return []
        if isinstance(result, FigureSpecList):
            return result.figures
        return []

    def render(
        self,
        specs: list[FigureSpec],
        statistics: dict[str, Any],
        output_dir: Path,
        figures_dir: Path,
        execution_id: str,
        source_data_ids: list[str],
    ) -> FigureRegistry:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        _configure_cjk_font(font_manager)

        figures_dir.mkdir(parents=True, exist_ok=True)
        registry = FigureRegistry()

        for index, spec in enumerate(specs, start=1):
            series = self._resolve_series(spec, statistics, output_dir)
            if not series:
                continue
            labels, values = series

            fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
            try:
                if spec.chart == "pie":
                    ax.pie(values, labels=labels, autopct="%1.1f%%")
                elif spec.chart == "line":
                    ax.plot(labels, values, marker="o")
                elif spec.chart == "scatter":
                    ax.scatter(labels, values)
                elif spec.chart == "histogram":
                    ax.hist(values, bins=min(20, max(3, len(values))))
                else:
                    ax.bar(labels, values)
                    ax.tick_params(axis="x", rotation=30)
                ax.set_title(spec.title)
                if spec.x_label:
                    ax.set_xlabel(spec.x_label)
                if spec.y_label:
                    ax.set_ylabel(spec.y_label)
                fig.tight_layout()
                artifact = figures_dir / f"fig{index}.png"
                fig.savefig(artifact)
            finally:
                plt.close(fig)

            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()[:16]
            try:
                figure_type = FigureType(spec.figure_type)
            except ValueError:
                figure_type = FigureType.COMPARISON

            record = FigureRecord(
                figure_id=f"FIG-{index:03d}",
                title=spec.title,
                caption=spec.caption or spec.title,
                figure_type=figure_type,
                source_data_ids=list(source_data_ids),
                source_execution_ids=[execution_id],
                generation_code=f"autopilot.figure({spec.chart})",
                artifact_path=str(artifact),
                hash=digest,
                verification_status="verified",
                metadata={
                    "chart": spec.chart,
                    "supported_claim": spec.supported_claim,
                    "source": spec.source,
                    "statistics_keys": spec.statistics_keys,
                    "output_file": spec.output_file,
                    "labels": [str(x) for x in labels],
                    "values": [float(v) for v in values],
                },
            )
            registry.register(record)

        return registry

    @staticmethod
    def _resolve_series(
        spec: FigureSpec,
        statistics: dict[str, Any],
        output_dir: Path,
    ) -> Optional[tuple[list[Any], list[float]]]:
        if spec.source == "output_file" and spec.output_file:
            path = output_dir / spec.output_file
            if not path.exists():
                return None
            headers, rows = read_output_table(path)
            columns = spec.output_columns or headers[:2]
            if len(columns) < 2:
                return None
            try:
                x_idx = headers.index(columns[0])
                y_idx = headers.index(columns[1])
            except ValueError:
                return None
            labels: list[Any] = []
            values: list[float] = []
            for row in rows:
                if x_idx >= len(row) or y_idx >= len(row):
                    continue
                try:
                    value = float(row[y_idx])
                except (TypeError, ValueError):
                    continue
                labels.append(row[x_idx])
                values.append(value)
            if not values:
                return None
            return labels, values

        keys = [k for k in spec.statistics_keys if k in statistics]
        numeric = [
            (k, float(statistics[k]))
            for k in keys
            if isinstance(statistics[k], (int, float))
            and not isinstance(statistics[k], bool)
        ]
        if not numeric:
            # Fall back to every numeric statistic, but only when a spec asked
            # for statistics and nothing else was usable.
            numeric = [
                (k, float(v))
                for k, v in statistics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
        if not numeric:
            return None
        # Labels and values must stay paired: a requested key may be non-numeric
        # and dropping it from only one of the two lists breaks the chart.
        labels = [k for k, _ in numeric]
        values = [v for _, v in numeric]
        return labels, values


def _configure_cjk_font(font_manager) -> None:
    """Pick an installed CJK font so Chinese labels render instead of boxes."""
    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
        "Source Han Sans SC", "SimSun", "Arial Unicode MS", "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            import matplotlib

            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


# ═══════════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════════

class TableBuilder:
    """Builds tables whose every numeric cell carries a source id."""

    def build_statistics_table(
        self,
        statistics: dict[str, Any],
        source_id: str,
        title: str = "求解结果统计",
    ) -> Optional[TableRecord]:
        rows: list[list[TableCell]] = []
        for key, value in statistics.items():
            if isinstance(value, dict):
                continue
            rows.append([
                TableCell(value=str(key), source_id=source_id, source_description="statistics key"),
                TableCell(value=value, source_id=source_id, source_description="executed program output"),
            ])
        if not rows:
            return None
        return TableRecord(
            table_id="TAB-001",
            title=title,
            table_type=TableType.OPTIMIZATION_RESULT,
            headers=["指标", "数值"],
            rows=rows,
            source_ids=[source_id],
            verification_status="verified",
        )

    def build_nested_statistics_table(
        self,
        statistics: dict[str, Any],
        source_id: str,
        table_id: str,
        title: str,
        key: str,
    ) -> Optional[TableRecord]:
        """Turn a nested dict statistic into a real table."""
        block = statistics.get(key)
        if not isinstance(block, dict) or not block:
            return None

        first = next((v for v in block.values() if isinstance(v, dict)), None)
        if first is None:
            headers = ["项目", "数值"]
            rows = [
                [
                    TableCell(value=str(k), source_id=source_id, source_description=f"statistics.{key}.key"),
                    TableCell(value=v, source_id=source_id, source_description=f"statistics.{key}.{k}"),
                ]
                for k, v in block.items()
            ]
        else:
            columns = list(first.keys())
            headers = ["类别"] + [str(c) for c in columns]
            rows = []
            for category, values in block.items():
                if not isinstance(values, dict):
                    continue
                row = [TableCell(value=str(category), source_id=source_id, source_description=f"statistics.{key}.{category}")]
                for column in columns:
                    row.append(TableCell(
                        value=values.get(column),
                        source_id=source_id,
                        source_description=f"statistics.{key}.{category}.{column}",
                    ))
                rows.append(row)

        return TableRecord(
            table_id=table_id,
            title=title,
            table_type=TableType.OPTIMIZATION_RESULT,
            headers=headers,
            rows=rows,
            source_ids=[source_id],
            verification_status="verified",
        )

    def build_output_table(
        self,
        output_dir: Path,
        file_name: str,
        table_id: str,
        title: str,
        source_id: str,
        max_rows: int = 30,
    ) -> Optional[TableRecord]:
        path = output_dir / file_name
        if not path.exists():
            return None
        headers, rows = read_output_table(path)
        if not headers:
            return None
        table_rows = [
            [
                TableCell(value=cell, source_id=source_id, source_description=f"{file_name}:row")
                for cell in row
            ]
            for row in rows[:max_rows]
        ]
        return TableRecord(
            table_id=table_id,
            title=title,
            table_type=TableType.DESCRIPTIVE_STATS,
            headers=headers,
            rows=table_rows,
            source_ids=[source_id],
            verification_status="verified",
            metadata={"truncated": len(rows) > max_rows, "total_rows": len(rows), "file": file_name},
        )


# ═══════════════════════════════════════════════════════════════
# Progressive paper writing
# ═══════════════════════════════════════════════════════════════

CUMCM_SECTIONS: list[tuple[str, str]] = [
    ("问题重述", "Restate the problem in your own words using only the parsed problem facts."),
    ("问题分析", "Analyse each subproblem: what is asked, what data exists, what approach is needed."),
    ("模型假设", "State the assumptions actually made, each tied to an ambiguity or data limitation."),
    ("符号说明", "Define the symbols used by the model."),
    ("模型建立与求解", "Present the mathematical model and the solving method that was actually executed."),
    ("结果分析", "Report the verified results and interpret them per subproblem."),
    ("模型检验", "Report the verification that was actually performed and its outcome."),
    ("模型评价与推广", "Evaluate strengths, weaknesses and applicability of the model."),
]


class SectionDraft(BaseModel):
    section_id: str = ""
    title: str
    paragraphs: list[str] = Field(default_factory=list)
    equations_latex: list[str] = Field(default_factory=list)
    used_figure_ids: list[str] = Field(default_factory=list)
    used_table_ids: list[str] = Field(default_factory=list)
    claim_texts: list[str] = Field(default_factory=list)
    blocked: bool = False
    blocked_reason: str = ""


class OutlineSection(BaseModel):
    title: str
    purpose: str = ""


class PaperOutline(BaseModel):
    title: str = ""
    abstract_points: list[str] = Field(default_factory=list)
    sections: list[OutlineSection] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class PaperWriter:
    """Writes the paper section by section from the verified context package."""

    def __init__(self, router: ModelRouter):
        self._router = router

    async def build_outline(self, package: PaperContextPackage) -> PaperOutline:
        prompt = "\n".join([
            "Create the outline of a Chinese CUMCM competition paper from the "
            "verified context package below.",
            "",
            "## VERIFIED CONTEXT PACKAGE",
            package.to_prompt(),
            "",
            "## RULES",
            "- Use this canonical section order:",
            "  " + " / ".join(t for t, _ in CUMCM_SECTIONS),
            "- `title` is the paper title in Chinese.",
            "- `abstract_points` lists the concrete points the abstract must state; "
            "each must be supported by the verified statistics above.",
            "- Never invent results that are not in the context package.",
        ])
        profile = TaskProfile.for_task_type(TaskType.PAPER_GENERATION)
        outline = await self._router.route_structured_generate(
            profile=profile,
            prompt=prompt,
            output_schema=PaperOutline,
            system_prompt=(
                "You are an experienced CUMCM paper author. You only write about "
                "results that exist in the verified context package."
            ),
        )
        if not isinstance(outline, PaperOutline):
            raise TypeError("Expected PaperOutline")
        if not outline.sections:
            outline.sections = [OutlineSection(title=t, purpose=p) for t, p in CUMCM_SECTIONS]
        return outline

    async def write_section(
        self,
        outline_section: OutlineSection,
        package: PaperContextPackage,
        index: int,
        correction: Optional[str] = None,
    ) -> SectionDraft:
        ids = package.available_ids()
        prompt = "\n".join([
            f"Write section {index}: 《{outline_section.title}》 of a Chinese CUMCM paper.",
            f"Section purpose: {outline_section.purpose}",
            "",
            "## VERIFIED CONTEXT PACKAGE (the ONLY information you may use)",
            package.to_prompt(),
            "",
            "## AVAILABLE IDS",
            f"figures: {ids['figure_ids']}",
            f"tables: {ids['table_ids']}",
            f"equations: {ids['equation_ids']}",
            "",
            "## RULES",
            "- Write in Chinese, formal academic style.",
            "- Every number you state MUST appear in `verified_statistics` or in the "
            "output files described above. Do not compute or estimate new numbers.",
            "- Reference figures/tables by their exact ids in `used_figure_ids` / "
            "`used_table_ids` and mention them in the text as 图/表.",
            "- Put display equations in `equations_latex` as raw LaTeX (no $$).",
            "- `claim_texts` lists the factual claims this section makes.",
            "- If the context package lacks the results this section needs, set "
            "`blocked=true`, explain in `blocked_reason`, and DO NOT invent content.",
        ])
        if correction:
            prompt += "\n\n## CORRECTION REQUIRED\n" + correction
        profile = TaskProfile.for_task_type(TaskType.PAPER_GENERATION)
        draft = await self._router.route_structured_generate(
            profile=profile,
            prompt=prompt,
            output_schema=SectionDraft,
            system_prompt=(
                "You are a rigorous CUMCM paper writer. You never fabricate numbers, "
                "never claim unverified results, and never cite figures that do not exist."
            ),
        )
        if not isinstance(draft, SectionDraft):
            raise TypeError("Expected SectionDraft")
        draft.title = outline_section.title
        draft.section_id = f"SEC-{index:03d}"
        return draft

    async def write_abstract(
        self,
        outline: PaperOutline,
        package: PaperContextPackage,
        drafts: list[SectionDraft],
        correction: Optional[str] = None,
    ) -> str:
        prompt = "\n".join([
            "Write the Chinese abstract (摘要) of a CUMCM paper, about 300-500 characters.",
            "",
            "## OUTLINE POINTS TO COVER",
            json.dumps(outline.abstract_points, ensure_ascii=False, indent=2),
            "",
            "## VERIFIED RESULTS (the only numbers you may state)",
            json.dumps(package.verified_statistics, ensure_ascii=False, indent=2),
            "",
            "## SECTION SUMMARIES",
            json.dumps(
                [{"title": d.title, "paragraphs": d.paragraphs[:1]} for d in drafts],
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Rules: no invented numbers, no claims beyond the verified results above.",
        ])
        if correction:
            prompt += "\n\n## CORRECTION REQUIRED\n" + correction
        profile = TaskProfile.for_task_type(TaskType.PAPER_GENERATION)

        class AbstractOut(BaseModel):
            abstract: str = ""

        result = await self._router.route_structured_generate(
            profile=profile,
            prompt=prompt,
            output_schema=AbstractOut,
            system_prompt="You write precise CUMCM abstracts grounded only in verified results.",
        )
        return getattr(result, "abstract", "")


def drafts_to_paper_ir(
    title: str,
    abstract: str,
    keywords: list[str],
    drafts: list[SectionDraft],
) -> PaperIR:
    sections: list[PaperSection] = []
    for draft in drafts:
        blocks: list[ContentBlock] = []
        for paragraph in draft.paragraphs:
            if paragraph.strip():
                blocks.append(ContentBlock(block_type=BlockType.PARAGRAPH, text=paragraph.strip()))
        for latex in draft.equations_latex:
            if latex.strip():
                blocks.append(ContentBlock(block_type=BlockType.EQUATION, text=latex.strip()))
        for figure_id in draft.used_figure_ids:
            blocks.append(ContentBlock(
                block_type=BlockType.FIGURE,
                text=figure_id,
                figure_ids=[figure_id],
            ))
        for table_id in draft.used_table_ids:
            blocks.append(ContentBlock(
                block_type=BlockType.TABLE,
                text=table_id,
                table_ids=[table_id],
            ))
        sections.append(PaperSection(
            section_id=draft.section_id or f"SEC-{len(sections) + 1:03d}",
            title=draft.title,
            purpose="",
            content_blocks=blocks,
            figure_ids=list(draft.used_figure_ids),
            table_ids=list(draft.used_table_ids),
        ))

    return PaperIR(
        title=title,
        abstract=abstract,
        keywords=keywords,
        sections=sections,
        references=[],
    )


# ═══════════════════════════════════════════════════════════════
# PDF build
# ═══════════════════════════════════════════════════════════════

def build_cumcm_latex(
    paper: PaperIR,
    figures: FigureRegistry,
    tables: TableRegistry,
    build_dir: Path,
) -> str:
    """Render a CUMCM-style Chinese LaTeX document with real figures/tables."""
    from mathmodel.paper.renderer import _check_latex_injection, _latex_escape

    lines = [
        r"\documentclass[11pt]{ctexart}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{geometry}",
        r"\usepackage{float}",
        r"\geometry{a4paper,margin=2.5cm}",
        r"\begin{document}",
        r"\begin{center}{\LARGE\bfseries " + _latex_escape(paper.title) + r"}\end{center}",
        r"\vspace{1em}",
    ]

    if paper.abstract:
        lines.append(r"\begin{center}{\large\bfseries 摘\quad 要}\end{center}")
        lines.append(_latex_escape(paper.abstract))
        lines.append("")
    if paper.keywords:
        lines.append(r"\noindent\textbf{关键词：} " + _latex_escape("，".join(paper.keywords)))
        lines.append(r"\newpage")

    for section in paper.sections:
        lines.append(r"\section{" + _latex_escape(section.title) + "}")
        for block in section.content_blocks:
            if block.block_type == BlockType.PARAGRAPH:
                lines.append(_latex_escape(block.text))
                lines.append("")
            elif block.block_type == BlockType.EQUATION:
                lines.append(r"\begin{equation}")
                lines.append(block.text)
                lines.append(r"\end{equation}")
            elif block.block_type == BlockType.FIGURE:
                # The block's structured reference is authoritative; the text is
                # only a caption, so fall back to it for older IR that stored the
                # id there instead.
                figure_id = (block.figure_ids or [block.text or ""])[0]
                figure = figures.get(figure_id)
                if figure is None:
                    continue
                source = Path(figure.artifact_path)
                target = build_dir / source.name
                # Copying a file onto itself raises on Windows, so only copy when
                # the figure is not already sitting in the build directory.
                if source.exists() and source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                lines.extend([
                    r"\begin{figure}[H]",
                    r"\centering",
                    r"\includegraphics[width=0.8\textwidth]{" + target.name + "}",
                    r"\caption{" + _latex_escape(figure.caption or figure.title) + "}",
                    r"\end{figure}",
                    "",
                ])
            elif block.block_type == BlockType.TABLE:
                table_id = (block.table_ids or [block.text or ""])[0]
                table = tables.get(table_id)
                if table is None:
                    continue
                column_spec = "|" + "l|" * len(table.headers) if table.headers else "|l|"
                lines.extend([
                    r"\begin{table}[H]",
                    r"\centering",
                    r"\caption{" + _latex_escape(table.title) + "}",
                    r"\begin{tabular}{" + column_spec + "}",
                    r"\hline",
                    " & ".join(_latex_escape(str(h)) for h in table.headers) + r" \\",
                    r"\hline",
                ])
                for row in table.rows:
                    cells = [_latex_escape("" if c.value is None else str(c.value)) for c in row]
                    lines.append(" & ".join(cells) + r" \\")
                    lines.append(r"\hline")
                lines.extend([r"\end{tabular}", r"\end{table}", ""])
            elif block.block_type == BlockType.LIST:
                lines.append(r"\begin{itemize}")
                for item in block.items:
                    lines.append(r"\item " + _latex_escape(item))
                lines.append(r"\end{itemize}")
                lines.append("")

    lines.append(r"\end{document}")
    tex = "\n".join(lines)

    issues = _check_latex_injection(tex)
    if issues:
        raise ValueError("LaTeX injection detected: " + "; ".join(issues))
    return tex


class PdfBuildResult(BaseModel):
    success: bool = False
    pdf_path: str = ""
    pdf_size: int = 0
    tex_path: str = ""
    command: str = ""
    exit_code: int = -1
    log_tail: str = ""
    compiler_available: bool = False
    pdf_header_ok: bool = False
    reason: str = ""


def build_pdf(
    tex_source: str,
    build_dir: Path,
    image: str = PAPER_COMPILE_IMAGE,
    timeout: int = 300,
) -> PdfBuildResult:
    """Compile LaTeX to PDF with xelatex in the paper image. Never fakes a PDF."""
    build_dir.mkdir(parents=True, exist_ok=True)
    tex_path = build_dir / "paper.tex"
    tex_path.write_text(tex_source, encoding="utf-8")

    result = PdfBuildResult(tex_path=str(tex_path))

    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        if probe.returncode != 0:
            result.reason = f"Docker image '{image}' is not available"
            return result
    except Exception as exc:
        result.reason = f"Docker unavailable: {exc}"
        return result

    result.compiler_available = True
    result.command = (
        f"docker run --rm -v {build_dir}:/work -w /work {image} "
        f"xelatex -interaction=nonstopmode paper.tex"
    )

    logs: list[str] = []
    for _ in range(2):  # twice so cross-references settle
        try:
            completed = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{build_dir}:/work",
                    "-w", "/work",
                    "--entrypoint", "xelatex",
                    image,
                    "-interaction=nonstopmode",
                    "paper.tex",
                ],
                capture_output=True, text=True, timeout=timeout,
                # xelatex writes UTF-8, and this host's locale is GBK. Without an
                # explicit encoding the reader thread raises UnicodeDecodeError and
                # leaves stdout as None, which then fails as "not subscriptable".
                encoding="utf-8", errors="replace",
            )
            logs.append((completed.stdout or "")[-4000:])
            result.exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            result.reason = "xelatex timed out"
            result.log_tail = "\n".join(logs)[-4000:]
            return result
        except Exception as exc:
            result.reason = str(exc)
            return result

    result.log_tail = "\n".join(logs)[-4000:]

    pdf_path = build_dir / "paper.pdf"
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        result.reason = "xelatex produced no PDF"
        return result

    result.pdf_path = str(pdf_path)
    result.pdf_size = pdf_path.stat().st_size
    result.pdf_header_ok = pdf_path.read_bytes()[:5] == b"%PDF-"
    result.success = result.pdf_header_ok
    if not result.success:
        result.reason = "Produced file is not a valid PDF"
    return result


# ═══════════════════════════════════════════════════════════════
# Support package
# ═══════════════════════════════════════════════════════════════

class SupportPackageBuilder:
    """Assembles output/ with paper.pdf, support/, and a manifest."""

    def __init__(self, output_dir: str | Path):
        self._output_dir = Path(output_dir)

    def build(
        self,
        paper_pdf: Optional[Path],
        paper_tex: Optional[Path],
        source_code: list[Path],
        processed_data: list[Path],
        original_files: list[Path],
        figures: list[Path],
        tables: list[Path],
        result_files: list[Path],
        manifest_extra: dict[str, Any],
    ) -> dict[str, Any]:
        out = self._output_dir
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        support = out / "support"
        dirs = {
            "source_code": support / "source_code",
            "data": support / "data_or_processed_data",
            "figures": support / "figures",
            "tables": support / "tables",
            "extra": support / "necessary_supporting_files",
        }
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        entries: dict[str, list[str]] = {k: [] for k in dirs}

        def copy_all(paths: list[Path], key: str, required: bool = False) -> None:
            for path in paths:
                path = Path(path)
                if not path.exists():
                    continue
                target = dirs[key] / path.name
                if target.exists():
                    target = dirs[key] / f"{path.stem}_{uuid4().hex[:4]}{path.suffix}"
                shutil.copy2(path, target)
                entries[key].append(str(target.relative_to(out)))

        if paper_pdf and Path(paper_pdf).exists():
            shutil.copy2(paper_pdf, out / "paper.pdf")
        if paper_tex and Path(paper_tex).exists():
            copy_all([Path(paper_tex)], "extra")

        copy_all(source_code, "source_code")
        copy_all(processed_data, "data")
        copy_all(original_files, "extra")
        copy_all(figures, "figures")
        copy_all(tables, "tables")
        copy_all(result_files, "data")

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper_pdf": "paper.pdf" if (out / "paper.pdf").exists() else None,
            "paper_pdf_size": (out / "paper.pdf").stat().st_size if (out / "paper.pdf").exists() else 0,
            "contents": entries,
            "counts": {k: len(v) for k, v in entries.items()},
            **manifest_extra,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
