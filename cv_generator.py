"""CV Generator — assembles tailored CV as Markdown and PDF."""
from __future__ import annotations

import io
from typing import List, Optional


# Section ordering by role type
_SECTION_ORDER: dict[str, List[str]] = {
    "data_analyst":      ["summary", "experience", "skills", "projects", "education", "certifications"],
    "business_analyst":  ["summary", "experience", "skills", "projects", "education", "certifications"],
    "data_scientist":    ["summary", "skills", "experience", "projects", "education", "certifications"],
    "ml_engineer":       ["summary", "skills", "projects", "experience", "education", "certifications"],
    "software_engineer": ["summary", "experience", "projects", "skills", "education", "certifications"],
    "product_manager":   ["summary", "experience", "projects", "skills", "education", "certifications"],
    "default":           ["summary", "experience", "skills", "projects", "education", "certifications"],
}


def get_section_order(role_type: str) -> List[str]:
    return _SECTION_ORDER.get(role_type, _SECTION_ORDER["default"])


# ---------------------------------------------------------------------------
# Markdown generator
# ---------------------------------------------------------------------------

def generate_cv_markdown(
    profile: dict,
    selected_experiences: List[dict],
    selected_projects: List[dict],
    relevant_skills: List[str],
    jd_analysis: dict,
    summary: str,
) -> str:
    personal = profile.get("personal", {})
    role_type = jd_analysis.get("role_type", "default")
    section_order = get_section_order(role_type)
    skills_data = profile.get("skills", {})

    lines: List[str] = []

    # ── Header ──
    name = personal.get("name") or "Your Name"
    lines.append(f"# {name}")

    contact_parts = [
        personal.get("email", ""),
        personal.get("phone", ""),
        personal.get("linkedin", ""),
        personal.get("github", ""),
        personal.get("location", ""),
    ]
    lines.append(" | ".join(p for p in contact_parts if p))
    lines.append("")

    # ── Section builders ──
    def _summary_lines() -> List[str]:
        if not summary:
            return []
        return ["## Summary", "", summary, ""]

    def _experience_lines() -> List[str]:
        if not selected_experiences:
            return []
        out = ["## Experience", ""]
        for exp in selected_experiences[:4]:  # max 4 roles
            title = exp.get("title", "")
            company = exp.get("company", "")
            if not title and not company:
                continue
            start = exp.get("start_date", "")
            end = exp.get("end_date", "Present")
            location = exp.get("location", "")

            header = f"**{title}** | {company}"
            if location:
                header += f" | {location}"
            out.append(header)
            out.append(f"*{start} – {end}*")
            bullets = [b for b in exp.get("bullets", []) if b and len(b.strip()) > 5]
            for bullet in bullets[:4]:  # max 4 bullets per role
                out.append(f"- {bullet}")
            out.append("")
        return out

    def _projects_lines() -> List[str]:
        if not selected_projects:
            return []
        out = ["## Projects", ""]
        for proj in selected_projects[:3]:  # max 3 projects
            name_p = proj.get("name", "")
            if not name_p:
                continue
            skill_tags = ", ".join(proj.get("skills", [])[:5])
            url = proj.get("url", "")

            header = f"**{name_p}**"
            if skill_tags:
                header += f" | *{skill_tags}*"
            if url:
                header += f" | {url}"
            out.append(header)
            bullets = [b for b in proj.get("bullets", []) if b and len(b.strip()) > 5]
            for bullet in bullets[:2]:  # max 2 bullets per project
                out.append(f"- {bullet}")
            out.append("")
        return out

    def _skills_lines() -> List[str]:
        technical = skills_data.get("technical", [])
        tools = skills_data.get("tools", [])
        soft = skills_data.get("soft", [])

        # Prioritize relevant skills, keep up to 14 total
        relevant_set = {s.lower() for s in relevant_skills}
        tech_out = sorted(technical, key=lambda s: (0 if s.lower() in relevant_set else 1))[:10]
        tools_out = sorted(tools, key=lambda s: (0 if s.lower() in relevant_set else 1))[:8]

        out = ["## Skills", ""]
        if tech_out:
            out.append(f"**Technical:** {', '.join(tech_out)}")
        if tools_out:
            out.append(f"**Tools:** {', '.join(tools_out)}")
        if soft:
            out.append(f"**Soft Skills:** {', '.join(soft[:4])}")
        out.append("")
        return out

    def _education_lines() -> List[str]:
        education = profile.get("education", [])
        if not education:
            return []
        out = ["## Education", ""]
        for edu in education:
            degree = edu.get("degree", "")
            field = edu.get("field", "")
            institution = edu.get("institution", "")
            start = edu.get("start_date", "")
            end = edu.get("end_date", "")
            gpa = edu.get("gpa", "")
            courses = edu.get("relevant_courses", [])

            out.append(f"**{degree} in {field}** | {institution}")
            if start or end:
                out.append(f"*{start} – {end}*")
            if gpa:
                out.append(f"GPA: {gpa}")
            if courses:
                out.append(f"Relevant Courses: {', '.join(courses)}")
            out.append("")
        return out

    def _certifications_lines() -> List[str]:
        _BAD = {"why ", "clarity", "leverage", "motivated", "i'm drawn", "i already",
                "0→1", "looking to", "i want to", "i am drawn"}

        def _ok(s: str) -> bool:
            if not isinstance(s, str) or len(s.strip()) < 3 or len(s) > 120:
                return False
            return not any(kw in s.lower() for kw in _BAD)

        certs = [c for c in profile.get("certifications", []) if _ok(c)]
        achievements = [a for a in profile.get("achievements", []) if _ok(a)]
        if not certs and not achievements:
            return []
        out = []
        if certs:
            out.append("## Certifications")
            out.append("")
            for c in certs[:5]:
                out.append(f"- {c}")
            out.append("")
        if achievements:
            out.append("## Achievements")
            out.append("")
            for a in achievements[:4]:
                out.append(f"- {a}")
            out.append("")
        return out

    section_map = {
        "summary": _summary_lines,
        "experience": _experience_lines,
        "projects": _projects_lines,
        "skills": _skills_lines,
        "education": _education_lines,
        "certifications": _certifications_lines,
    }

    for section in section_order:
        if section in section_map:
            lines.extend(section_map[section]())

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF generator (ReportLab) — professional layout matching reference CV
# ---------------------------------------------------------------------------

def generate_pdf_bytes(
    profile: dict,
    selected_experiences: List[dict],
    selected_projects: List[dict],
    relevant_skills: List[str],
    jd_analysis: dict,
    summary: str,
) -> Optional[bytes]:
    """Generate a professionally styled PDF matching the reference CV layout."""
    try:
        import re as _re
        from reportlab.lib import colors  # type: ignore
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT  # type: ignore
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
        from reportlab.lib.units import cm, mm  # type: ignore
        from reportlab.platypus import (  # type: ignore
            HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        return None

    personal = profile.get("personal", {})
    role_type = jd_analysis.get("role_type", "default")
    section_order = get_section_order(role_type)
    skills_data = profile.get("skills", {})

    # ── Colour palette (matches reference) ──
    DARK    = colors.HexColor("#1a1a1a")
    ACCENT  = colors.HexColor("#1a3a5c")   # dark navy — section headers & company names
    GRAY    = colors.HexColor("#555555")
    LGRAY   = colors.HexColor("#888888")
    RULE    = colors.HexColor("#d0d0d0")

    PAGE_W = A4[0]
    LM = RM = 1.7 * cm
    CONTENT_W = PAGE_W - LM - RM

    _styles = getSampleStyleSheet()

    # ── Type styles — tuned for 1-page fit ──
    s_name = ParagraphStyle("s_name", parent=_styles["Normal"],
        fontSize=22, fontName="Helvetica-Bold", leading=26,
        spaceAfter=1, alignment=TA_CENTER, textColor=DARK)

    s_tagline = ParagraphStyle("s_tagline", parent=_styles["Normal"],
        fontSize=9.5, fontName="Helvetica", leading=13,
        spaceAfter=2, alignment=TA_CENTER, textColor=ACCENT)

    s_contact = ParagraphStyle("s_contact", parent=_styles["Normal"],
        fontSize=8, fontName="Helvetica", leading=11,
        spaceAfter=5, alignment=TA_CENTER, textColor=GRAY)

    s_h2 = ParagraphStyle("s_h2", parent=_styles["Normal"],
        fontSize=8, fontName="Helvetica-Bold", leading=10,
        spaceBefore=4, spaceAfter=2, textColor=ACCENT)

    s_job_left = ParagraphStyle("s_job_left", parent=_styles["Normal"],
        fontSize=9, fontName="Helvetica-Bold", leading=11,
        spaceAfter=1, textColor=DARK)

    s_job_right = ParagraphStyle("s_job_right", parent=_styles["Normal"],
        fontSize=8, fontName="Helvetica", leading=11,
        spaceAfter=1, alignment=TA_RIGHT, textColor=GRAY)

    s_tech = ParagraphStyle("s_tech", parent=_styles["Normal"],
        fontSize=7.5, fontName="Helvetica-Oblique", leading=10,
        spaceAfter=1, textColor=LGRAY)

    s_bullet = ParagraphStyle("s_bullet", parent=_styles["Normal"],
        fontSize=8.5, fontName="Helvetica", leading=11,
        spaceAfter=2, leftIndent=10, firstLineIndent=-8, textColor=DARK)

    s_body = ParagraphStyle("s_body", parent=_styles["Normal"],
        fontSize=8.5, fontName="Helvetica", leading=12,
        spaceAfter=3, textColor=DARK)

    s_skills_line = ParagraphStyle("s_skills_line", parent=_styles["Normal"],
        fontSize=8.5, fontName="Helvetica", leading=12,
        spaceAfter=2, textColor=DARK)

    def _esc(text: str) -> str:
        """Escape special XML chars for ReportLab paragraphs."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _md(text: str) -> str:
        """Convert **bold** and *italic* markdown to ReportLab HTML."""
        text = _esc(text)
        text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = _re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        return text

    def _hr(thickness: float = 0.5, color=RULE, space_before: float = 1, space_after: float = 3):
        return HRFlowable(width="100%", thickness=thickness, color=color,
                          spaceBefore=space_before, spaceAfter=space_after)

    def _section_header(title: str):
        """Small-caps style header with coloured rule below."""
        return [
            Spacer(1, 2),
            Paragraph(title.upper(), s_h2),
            _hr(thickness=0.8, color=ACCENT, space_before=1, space_after=4),
        ]

    def _entry_header(left_html: str, right_html: str):
        """Two-column table row: bold left (title/company), grey right (location · date)."""
        tbl = Table(
            [[Paragraph(left_html, s_job_left), Paragraph(right_html, s_job_right)]],
            colWidths=[CONTENT_W * 0.68, CONTENT_W * 0.32],
        )
        tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        return tbl

    def _bullet(text: str):
        return Paragraph(f"› {_md(text)}", s_bullet)

    story: list = []

    # ══ HEADER ══════════════════════════════════════════════════════════════
    name = personal.get("name") or "Your Name"
    story.append(Paragraph(_esc(name), s_name))

    tagline = personal.get("tagline") or personal.get("headline") or personal.get("title") or ""
    if tagline:
        story.append(Paragraph(_esc(tagline), s_tagline))

    # Build contact line: email · phone · linkedin · GitHub: github · location
    contact_items = []
    if personal.get("email"):
        contact_items.append(_esc(personal["email"]))
    if personal.get("phone"):
        contact_items.append(_esc(personal["phone"]))
    if personal.get("linkedin"):
        contact_items.append(_esc(personal["linkedin"]))
    if personal.get("github"):
        contact_items.append(f"GitHub: {_esc(personal['github'])}")
    if personal.get("location"):
        contact_items.append(_esc(personal["location"]))
    if contact_items:
        story.append(Paragraph("  ·  ".join(contact_items), s_contact))

    story.append(_hr(thickness=1.0, color=DARK, space_before=2, space_after=6))

    # ══ SECTION BUILDERS ═══════════════════════════════════════════════════

    def _add_summary():
        if not summary:
            return
        story.extend(_section_header("Profile"))
        story.append(Paragraph(_md(summary), s_body))
        story.append(Spacer(1, 3))

    def _add_experience():
        if not selected_experiences:
            return
        story.extend(_section_header("Experience"))
        for exp in selected_experiences[:4]:  # hard cap: max 4 roles
            title    = _esc(exp.get("title", ""))
            company  = _esc(exp.get("company", ""))
            start    = _esc(exp.get("start_date", ""))
            end      = _esc(exp.get("end_date", "Present"))
            location = _esc(exp.get("location", ""))

            if not title and not company:
                continue  # skip empty/junk entries

            left  = f"<b>{title}</b> — <font color='#{ACCENT.hexval()[2:]}'>{company}</font>"
            right_parts = [p for p in [location, f"{start} – {end}"] if p]
            right = "  ·  ".join(right_parts)

            story.append(_entry_header(left, right))
            bullets = [b for b in exp.get("bullets", []) if b and len(b.strip()) > 5]
            for b in bullets[:4]:  # hard cap: max 4 bullets per role
                story.append(_bullet(b))
            story.append(Spacer(1, 3))

    def _add_projects():
        if not selected_projects:
            return
        story.extend(_section_header("Projects"))
        for proj in selected_projects[:3]:  # hard cap: max 3 projects
            name_p   = _esc(proj.get("name", ""))
            desc     = _esc(proj.get("description", "") or proj.get("subtitle", ""))
            url      = _esc(proj.get("url", ""))
            skills_p = proj.get("skills", [])[:6]
            tech_str = "  ·  ".join(_esc(s) for s in skills_p)

            if not name_p:
                continue

            # Header line: Name · descriptor · url (colored)
            header_parts = [f"<b>{name_p}</b>"]
            if desc:
                header_parts.append(desc)
            if url:
                header_parts.append(f"<font color='#{ACCENT.hexval()[2:]}'>{url}</font>")
            story.append(Paragraph("  ·  ".join(header_parts), s_job_left))
            if tech_str:
                story.append(Paragraph(tech_str, s_tech))
            bullets = [b for b in proj.get("bullets", []) if b and len(b.strip()) > 5]
            for b in bullets[:2]:  # hard cap: max 2 bullets per project
                story.append(_bullet(b))
            story.append(Spacer(1, 3))

    def _add_skills():
        technical = skills_data.get("technical", [])
        tools     = skills_data.get("tools", [])
        soft      = skills_data.get("soft", [])
        languages = skills_data.get("languages", [])

        relevant_set = {s.lower() for s in relevant_skills}
        tech_out  = sorted(technical, key=lambda s: (0 if s.lower() in relevant_set else 1))[:12]
        tools_out = sorted(tools,     key=lambda s: (0 if s.lower() in relevant_set else 1))[:8]

        if not tech_out and not tools_out and not soft:
            return
        story.extend(_section_header("Skills"))
        if tech_out:
            story.append(Paragraph(
                f"<b>Technical:</b> {', '.join(_esc(s) for s in tech_out)}", s_skills_line))
        if tools_out:
            story.append(Paragraph(
                f"<b>Tools:</b> {', '.join(_esc(s) for s in tools_out)}", s_skills_line))
        if soft:
            story.append(Paragraph(
                f"<b>Soft Skills:</b> {', '.join(_esc(s) for s in soft[:5])}", s_skills_line))
        if languages:
            story.append(Paragraph(
                f"<b>Languages:</b> {', '.join(_esc(s) for s in languages)}", s_skills_line))
        story.append(Spacer(1, 3))

    def _add_education():
        education = profile.get("education", [])
        if not education:
            return
        story.extend(_section_header("Education"))
        for edu in education:
            degree  = _esc(edu.get("degree", ""))
            field   = _esc(edu.get("field", ""))
            inst    = _esc(edu.get("institution", ""))
            start   = _esc(edu.get("start_date", ""))
            end     = _esc(edu.get("end_date", ""))
            gpa     = _esc(edu.get("gpa", ""))
            courses = edu.get("relevant_courses", [])
            location = _esc(edu.get("location", ""))

            deg_str = f"<b>{degree}{' in ' + field if field else ''}</b> — " \
                      f"<font color='#{ACCENT.hexval()[2:]}'>{inst}</font>"
            right_parts = [p for p in [location, f"{start} – {end}" if (start or end) else ""] if p]
            right = "  ·  ".join(right_parts)
            story.append(_entry_header(deg_str, right))

            sub_parts = []
            if gpa:
                sub_parts.append(f"Grade: {gpa}")
            if courses:
                sub_parts.append("  ·  ".join(_esc(c) for c in courses[:3]))
            if sub_parts:
                story.append(Paragraph("  ·  ".join(sub_parts), s_tech))
            story.append(Spacer(1, 3))

    def _add_certifications():
        _BAD_KEYWORDS = {"why ", "clarity", "leverage", "motivated", "i'm drawn", "i already", "0→1",
                         "looking to", "i want to", "environment", "i am drawn"}

        def _is_real_cert(s: str) -> bool:
            """Filter out WHY-COMPANY paragraphs and long junk masquerading as achievements."""
            if not isinstance(s, str) or len(s.strip()) < 3:
                return False
            if len(s) > 120:  # real cert names are short
                return False
            low = s.lower()
            if any(kw in low for kw in _BAD_KEYWORDS):
                return False
            return True

        certs        = [c for c in profile.get("certifications", []) if _is_real_cert(c)]
        achievements = [a for a in profile.get("achievements", []) if _is_real_cert(a)]

        if certs:
            story.extend(_section_header("Certifications"))
            for c in certs[:5]:
                story.append(_bullet(c))
            story.append(Spacer(1, 3))
        if achievements:
            story.extend(_section_header("Achievements"))
            for a in achievements[:4]:
                story.append(_bullet(a))
            story.append(Spacer(1, 3))

    # ── Assemble in role-type order ──────────────────────────────────────
    section_fn = {
        "summary":        _add_summary,
        "experience":     _add_experience,
        "projects":       _add_projects,
        "skills":         _add_skills,
        "education":      _add_education,
        "certifications": _add_certifications,
    }

    for section in section_order:
        if section in section_fn:
            section_fn[section]()

    # ── Build PDF ────────────────────────────────────────────────────────
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=RM, leftMargin=LM,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm,
        title=name,
        author=name,
    )
    doc.build(story)
    return buffer.getvalue()
