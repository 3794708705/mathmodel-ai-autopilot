"""MathModel AI — CUMCM Autopilot: Problem Intake.

Reads the uploaded problem statement and every attachment for real.
Nothing here is inferred from a filename alone: PDF text is extracted,
Excel sheets are read cell by cell, CSV files are profiled, and the
structured ProblemContext is assembled from that parsed content.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from mathmodel.domain.data import ColumnProfile, FileRecord, ParserStatus
from mathmodel.files.parsers import CsvParser, ExcelParser, PdfParser, TxtParser
from mathmodel.files.service import FileService

STATEMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc"}
TABLE_EXTENSIONS = {".xlsx", ".xls", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

TEMPLATE_NAME_RE = re.compile(r"^\s*(result|结果)\s*\d+", re.IGNORECASE)
OUTPUT_FILE_RE = re.compile(r"(result\s*\d+\.xlsx)", re.IGNORECASE)
ATTACHMENT_MENTION_RE = re.compile(r"附件\s*([0-9０-９]+)")
ATTACHMENT_FILE_RE = re.compile(r"附件\s*([0-9０-９]+)")


class SheetSchema(BaseModel):
    """Real schema of one parsed sheet."""

    name: str
    headers: list[str] = Field(default_factory=list)
    row_count: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)
    sample_rows: list[list[Any]] = Field(default_factory=list)


class AttachmentInfo(BaseModel):
    """One uploaded file, as actually read."""

    file_id: str = ""
    original_name: str
    role: str = "other"  # problem_statement | data | template | image | other
    media_type: str = ""
    extension: str = ""
    size_bytes: int = 0
    sha256: str = ""
    storage_path: str = ""
    parser_name: Optional[str] = None
    parser_status: ParserStatus = ParserStatus.PENDING
    text_char_count: int = 0
    extracted_text: str = ""
    is_image_only: bool = False
    sheets: list[SheetSchema] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProblemContext(BaseModel):
    """Structured understanding of what was uploaded."""

    problem_text: str = ""
    problem_file_names: list[str] = Field(default_factory=list)
    attachments: list[AttachmentInfo] = Field(default_factory=list)
    data_file_names: list[str] = Field(default_factory=list)
    template_file_names: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    declared_attachments: list[str] = Field(default_factory=list)
    missing_attachments: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def attachment(self, name: str) -> Optional[AttachmentInfo]:
        for a in self.attachments:
            if a.original_name == name:
                return a
        return None

    def data_sheets(self) -> list[tuple[AttachmentInfo, SheetSchema]]:
        out = []
        for a in self.attachments:
            if a.role == "data":
                for sheet in a.sheets:
                    out.append((a, sheet))
        return out

    def to_prompt(self, max_sheet_rows: int = 5) -> str:
        """Compact, factual description used by downstream LLM stages."""
        parts = ["## PARSED ATTACHMENTS"]
        if not self.attachments:
            parts.append("(none)")
        for a in self.attachments:
            parts.append(
                f"- {a.original_name} [{a.role}] parser={a.parser_name} "
                f"status={a.parser_status.value} size={a.size_bytes}B"
            )
            for sheet in a.sheets:
                parts.append(
                    f"    sheet '{sheet.name}': {sheet.row_count} rows, "
                    f"columns={sheet.headers}"
                )
                for row in sheet.sample_rows[:max_sheet_rows]:
                    parts.append(f"      {row}")
        if self.required_outputs:
            parts.append("## REQUIRED OUTPUT FILES (from problem statement)")
            for name in self.required_outputs:
                parts.append(f"- {name}")
        if self.missing_attachments:
            parts.append("## ATTACHMENTS MENTIONED BUT NOT UPLOADED")
            for name in self.missing_attachments:
                parts.append(f"- {name}")
        return "\n".join(parts)


class ProblemIntake:
    """Ingests uploads and produces a parsed ProblemContext."""

    def __init__(self, storage_dir: str | Path):
        self._storage_dir = Path(storage_dir)
        self._service = FileService(self._storage_dir)

    # ── Public API ────────────────────────────────────────────

    def ingest(self, files: list[str | Path]) -> tuple[ProblemContext, list[FileRecord]]:
        records: list[FileRecord] = []
        attachments: list[AttachmentInfo] = []

        for path in files:
            record = self._service.ingest(path)
            records.append(record)
            attachments.append(self._read(record))

        context = self._build_context(attachments)
        return context, records

    def export_processed_data(self, context: ProblemContext, out_dir: str | Path) -> list[str]:
        """Write every parsed data sheet to CSV so the solver reads real data."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        for info in context.attachments:
            if info.role != "data":
                continue
            source = Path(info.storage_path)
            if info.extension == ".csv":
                target = out_dir / Path(info.original_name).name
                target.write_text(
                    source.read_text(encoding="utf-8", errors="replace"),
                    encoding="utf-8",
                )
                written.append(str(target))
                continue
            if info.extension in (".xlsx", ".xls"):
                from mathmodel.domain.data import FileRecord as _FR

                sheets = ExcelParser.read_sheets(
                    _FR(original_name=info.original_name, storage_path=info.storage_path)
                )
                for sheet_name, data in sheets.items():
                    stem = Path(info.original_name).stem
                    suffix = f"__{sheet_name}" if len(sheets) > 1 else ""
                    target = out_dir / f"{stem}{suffix}.csv"
                    with open(target, "w", encoding="utf-8", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(data["headers"])
                        for row in data["rows"]:
                            writer.writerow(["" if v is None else v for v in row])
                    written.append(str(target))

        return written

    # ── Reading ───────────────────────────────────────────────

    def _read(self, record: FileRecord) -> AttachmentInfo:
        info = AttachmentInfo(
            file_id=record.file_id,
            original_name=record.original_name,
            extension=record.extension,
            media_type=record.media_type,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            storage_path=record.storage_path,
            parser_name=record.parser_name,
        )

        ext = record.extension.lower()
        if ext == ".pdf":
            parsed = PdfParser.parse(record)
            pages = parsed.get("pages", [])
            text = "\n".join(p.get("text", "") for p in pages)
            info.extracted_text = text
            info.text_char_count = len(text)
            info.is_image_only = bool(parsed.get("is_image_only"))
            info.parser_name = parsed.get("parser", "pdf")
            info.parser_status = (
                ParserStatus.OCR_REQUIRED
                if info.is_image_only
                else ParserStatus.COMPLETED
            )
            info.notes.append(f"pages={parsed.get('page_count', 0)}")
            info.notes.append(f"text_chars={len(text)}")
            if parsed.get("error"):
                info.notes.append(f"error={parsed['error']}")
            return info

        if ext in (".txt", ".md"):
            parsed = TxtParser.parse(record)
            info.parser_status = ParserStatus.COMPLETED
            info.extracted_text = parsed.get("content", "")
            info.text_char_count = parsed.get("char_count", 0)
            return info

        if ext in (".xlsx", ".xls"):
            try:
                sheets = ExcelParser.read_sheets(record)
            except Exception as exc:  # unreadable workbook is reported, not guessed
                info.parser_status = ParserStatus.FAILED
                info.notes.append(f"excel_error={exc}")
                return info

            for sheet_name, data in sheets.items():
                headers = data["headers"]
                rows = data["rows"]
                profiles = [
                    CsvParser._profile_column(
                        idx, header, [CsvParser._cell_to_str(r, idx) for r in rows]
                    )
                    for idx, header in enumerate(headers)
                ]
                info.sheets.append(SheetSchema(
                    name=sheet_name,
                    headers=headers,
                    row_count=len(rows),
                    columns=profiles,
                    sample_rows=[
                        [("" if v is None else v) for v in row]
                        for row in rows[:5]
                    ],
                ))
            info.parser_status = ParserStatus.COMPLETED
            info.notes.append(f"sheets={len(info.sheets)}")
            return info

        if ext == ".csv":
            table, profile = CsvParser.parse(record)
            info.sheets.append(SheetSchema(
                name=Path(record.original_name).stem,
                headers=[c.name for c in profile.columns],
                row_count=profile.row_count,
                columns=profile.columns,
                sample_rows=[],
            ))
            info.parser_status = ParserStatus.COMPLETED
            return info

        if ext in IMAGE_EXTENSIONS:
            info.parser_status = ParserStatus.MULTIMODAL_REQUIRED
            info.notes.append("image attachment registered; OCR not performed")
            return info

        info.parser_status = ParserStatus.UNSUPPORTED
        info.notes.append(f"no parser for extension {ext}")
        return info

    # ── Context assembly ──────────────────────────────────────

    def _build_context(self, attachments: list[AttachmentInfo]) -> ProblemContext:
        context = ProblemContext()
        context.attachments = attachments

        texts: list[str] = []
        # A statement is a document carrying prose, classified by parsed content
        # rather than by file name. Prefer substantial documents, but if the
        # upload contains prose and none of it is long, still treat the longest
        # candidate as the statement instead of silently reading nothing.
        candidates = [
            info for info in attachments
            if info.extension in STATEMENT_EXTENSIONS
            and len(info.extracted_text.strip()) > 0
        ]
        statements = [i for i in candidates if len(i.extracted_text) > 200]
        if not statements and candidates:
            statements = [max(candidates, key=lambda i: len(i.extracted_text))]

        for info in statements:
            info.role = "problem_statement"
            context.problem_file_names.append(info.original_name)
            texts.append(info.extracted_text)

        for info in attachments:
            if info.role == "problem_statement":
                continue
            if info.extension in TABLE_EXTENSIONS:
                info.role = "template" if TEMPLATE_NAME_RE.match(info.original_name) else "data"
            elif info.extension in IMAGE_EXTENSIONS:
                info.role = "image"
            else:
                info.role = "other"

        context.problem_text = "\n\n".join(texts).strip()
        context.data_file_names = [
            a.original_name for a in attachments if a.role == "data"
        ]
        context.template_file_names = [
            a.original_name for a in attachments if a.role == "template"
        ]

        if context.problem_text:
            context.required_outputs = _unique(
                m.group(1).strip()
                for m in OUTPUT_FILE_RE.finditer(context.problem_text)
            )
            context.declared_attachments = _unique(
                f"附件{m.group(1)}"
                for m in ATTACHMENT_MENTION_RE.finditer(context.problem_text)
            )

        uploaded = {a.original_name for a in attachments}
        for declared in context.declared_attachments:
            index = re.sub(r"\D", "", declared)
            if not index:
                continue
            present = any(
                f"附件{index}" in name or f"附件 {index}" in name
                for name in uploaded
            )
            if not present:
                context.missing_attachments.append(declared)

        for info in attachments:
            if info.parser_status == ParserStatus.FAILED:
                context.warnings.append(
                    f"{info.original_name}: parse failed ({'; '.join(info.notes)})"
                )
            if info.parser_status == ParserStatus.OCR_REQUIRED:
                context.warnings.append(
                    f"{info.original_name}: no extractable text (likely a scan)"
                )

        return context


def _unique(values) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if value not in seen:
            seen[value] = None
    return list(seen)


def build_clarification_questions(context: ProblemContext) -> list[Any]:
    """Ask only about information that is genuinely missing.

    Returns an empty list when the upload is already sufficient. Blocking
    questions stop the run; non-blocking ones are recorded and the run
    continues, so ordinary gaps never stall the workflow.
    """
    from mathmodel.autopilot.state import ClarificationQuestion

    questions: list[ClarificationQuestion] = []

    if not context.problem_text.strip():
        questions.append(ClarificationQuestion(
            question_id="Q-STATEMENT",
            question=(
                "无法从上传的文件中提取赛题正文（PDF 可能是扫描件或图片）。"
                "请提供包含可复制文本的赛题文件，或直接粘贴赛题正文。"
            ),
            reason="赛题正文为空，后续问题理解无法进行",
            blocking=True,
        ))

    if context.missing_attachments:
        # Without any data file the modeling cannot start; with data files the
        # gap is worth reporting but must not stop the run.
        blocking = not context.data_file_names
        questions.append(ClarificationQuestion(
            question_id="Q-ATTACHMENTS",
            question=(
                "赛题正文提到了以下附件，但未检测到同名文件："
                + "、".join(context.missing_attachments)
                + "。"
                + (
                    "当前也没有任何数据类附件，无法开始建模，请补充上传。"
                    if blocking
                    else "如无需补充，可忽略；系统将按现有材料继续。"
                )
            ),
            reason="附件缺失可能影响数据与结果文件格式",
            blocking=blocking,
            options=["补充上传", "确认没有更多附件，按现有材料继续"],
        ))

    return questions
