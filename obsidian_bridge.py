"""
obsidian_bridge.py
==================
Gives Claude Code the ability to read from and write to the
PF-WRP Obsidian vault directly from the project directory.

Place this file in your Debris_Flow_App/ project root.

Claude Code will call these functions automatically per the
rules defined in CLAUDE.md.

Usage (Claude Code will run these — you don't call them manually):
    python obsidian_bridge.py --action read_context
    python obsidian_bridge.py --action log_session --date 2026-04-12
    python obsidian_bridge.py --action update_tasks --task "Download USGS CSV" --done
    python obsidian_bridge.py --action append_note --file "Science/Validation_Statistics.md" --content "..."
    python obsidian_bridge.py --action new_note --folder "Science" --title "Woolsey_Fire_Hindcast"
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURATION
# Update VAULT_PATH to match where your Obsidian vault lives.
# ============================================================
VAULT_PATH = Path.home() / "Desktop" / "PF-WRP Research Hub" / "PF-WRP Research Hub"

PATHS = {
    "moc":      VAULT_PATH / "🗺️ MOC — Map of Content.md",
    "context":  VAULT_PATH / "Sessions" / "Claude_Context_Template.md",
    "sessions": VAULT_PATH / "Sessions",
    "science":  VAULT_PATH / "Science",
    "code":     VAULT_PATH / "Code",
    "academic": VAULT_PATH / "Academic",
    "data":     VAULT_PATH / "Data",
}

SESSION_TEMPLATE = """# Session — {date}
> [[🗺️ MOC — Map of Content]] | Type: Session Log

## What Was Built
{built}

## Key Findings
{findings}

## Decisions Made
{decisions}

## Code Changes
- Files modified: {files}
- Summary: {code_summary}

## Open Tasks Created
{tasks}

## Next Session Should Start With
Paste [[Claude_Context_Template]] then:
{next_steps}
"""


# ============================================================
# CORE FUNCTIONS
# ============================================================

def read_context() -> str:
    """
    Reads the Claude Context Template note from the vault.
    Claude Code calls this at the start of every session to
    load the current project state without user re-explanation.
    """
    path = PATHS["context"]
    if not path.exists():
        return f"ERROR: Context note not found at {path}"
    content = path.read_text(encoding="utf-8")
    print(f"[obsidian_bridge] Loaded context from: {path}")
    print("=" * 60)
    print(content)
    print("=" * 60)
    return content


def read_moc() -> str:
    """
    Reads the Map of Content note to get current task status.
    """
    path = PATHS["moc"]
    if not path.exists():
        return f"ERROR: MOC not found at {path}"
    content = path.read_text(encoding="utf-8")
    print(f"[obsidian_bridge] Loaded MOC from: {path}")
    return content


def log_session(
    date: str = None,
    built: str = "- (fill in)",
    findings: str = "- (fill in)",
    decisions: str = "- (fill in)",
    files: str = "(fill in)",
    code_summary: str = "(fill in)",
    tasks: str = "- [ ] (fill in)",
    next_steps: str = "- (fill in)"
) -> str:
    """
    Creates a new dated session note in the vault's Sessions/ folder.
    Claude Code calls this automatically at the end of every session.

    Parameters
    ----------
    date         : str   Date string YYYY-MM-DD (defaults to today)
    built        : str   Bullet list of what was built this session
    findings     : str   Key scientific or technical findings
    decisions    : str   Architectural or methodological decisions made
    files        : str   Comma-separated list of modified files
    code_summary : str   Brief description of code changes
    tasks        : str   New open tasks as markdown checkboxes
    next_steps   : str   What to tell Claude at the start of next session

    Returns
    -------
    str  Path to the created session note
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    filename = f"{date}_Session.md"
    output_path = PATHS["sessions"] / filename

    if output_path.exists():
        # Don't overwrite — append with separator
        existing = output_path.read_text(encoding="utf-8")
        new_entry = SESSION_TEMPLATE.format(
            date=date + " (update)",
            built=built, findings=findings, decisions=decisions,
            files=files, code_summary=code_summary,
            tasks=tasks, next_steps=next_steps
        )
        output_path.write_text(existing + "\n\n---\n\n" + new_entry, encoding="utf-8")
        print(f"[obsidian_bridge] Appended to existing session note: {output_path}")
    else:
        content = SESSION_TEMPLATE.format(
            date=date,
            built=built, findings=findings, decisions=decisions,
            files=files, code_summary=code_summary,
            tasks=tasks, next_steps=next_steps
        )
        output_path.write_text(content, encoding="utf-8")
        print(f"[obsidian_bridge] Created session note: {output_path}")

    return str(output_path)


def append_to_note(relative_path: str, content: str, section_header: str = None) -> str:
    """
    Appends content to an existing note in the vault.
    Never overwrites — always appends to the bottom or to a named section.

    Parameters
    ----------
    relative_path  : str  Path relative to vault root, e.g. "Science/Validation_Statistics.md"
    content        : str  Markdown content to append
    section_header : str  If provided, inserts under this ## header (optional)

    Returns
    -------
    str  Confirmation message
    """
    target = VAULT_PATH / relative_path

    if not target.exists():
        return f"ERROR: Note not found: {target}\nUse new_note() to create it first."

    existing = target.read_text(encoding="utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_block = f"\n\n---\n*Updated: {timestamp}*\n\n{content}"

    target.write_text(existing + new_block, encoding="utf-8")
    print(f"[obsidian_bridge] Appended to: {target}")
    return f"Successfully appended to {relative_path}"


def create_new_note(folder: str, title: str, content: str = "") -> str:
    """
    Creates a brand new note in the specified vault folder.

    Parameters
    ----------
    folder  : str  Folder name: "Science", "Code", "Academic", "Data", "Sessions"
    title   : str  Note filename without .md extension
    content : str  Initial markdown content for the note

    Returns
    -------
    str  Path to the created note
    """
    if folder not in PATHS:
        return f"ERROR: Unknown folder '{folder}'. Valid: {list(PATHS.keys())}"

    folder_path = PATHS[folder]
    note_path = folder_path / f"{title}.md"

    if note_path.exists():
        return f"ERROR: Note already exists: {note_path}\nUse append_to_note() to add content."

    # Add backlink to MOC at top of every new note
    header = f"# {title.replace('_', ' ')}\n> [[🗺️ MOC — Map of Content]] | Type: {folder.capitalize()}\n\n---\n\n"
    note_path.write_text(header + content, encoding="utf-8")
    print(f"[obsidian_bridge] Created new note: {note_path}")
    return str(note_path)


def update_task_in_moc(task_text: str, mark_done: bool = True) -> str:
    """
    Marks a task as complete (or incomplete) in the MOC note.
    Searches for the task text and toggles its checkbox.

    Parameters
    ----------
    task_text : str   Partial or full text of the task to find
    mark_done : bool  True = check off, False = uncheck

    Returns
    -------
    str  Confirmation or error message
    """
    moc_path = PATHS["moc"]
    if not moc_path.exists():
        return f"ERROR: MOC not found at {moc_path}"

    content = moc_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    found = False

    for i, line in enumerate(lines):
        if task_text.lower() in line.lower():
            if mark_done and "- [ ]" in line:
                lines[i] = line.replace("- [ ]", "- [x]")
                found = True
                print(f"[obsidian_bridge] Checked off task: {line.strip()}")
            elif not mark_done and "- [x]" in line:
                lines[i] = line.replace("- [x]", "- [ ]")
                found = True
                print(f"[obsidian_bridge] Unchecked task: {line.strip()}")

    if not found:
        return f"Task containing '{task_text}' not found or already in target state."

    moc_path.write_text("\n".join(lines), encoding="utf-8")
    return f"MOC updated successfully."


def update_context_template(section: str, new_content: str) -> str:
    """
    Updates a specific section in the Claude Context Template note.
    Use this to keep the context current after completing tasks.

    Parameters
    ----------
    section     : str  Section header text to find and update
    new_content : str  New content to replace that section's content

    Returns
    -------
    str  Confirmation message
    """
    context_path = PATHS["context"]
    if not context_path.exists():
        return f"ERROR: Context note not found at {context_path}"

    content = context_path.read_text(encoding="utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Append update log at bottom rather than surgically editing
    update_block = (
        f"\n\n---\n"
        f"*Section updated {timestamp}: {section}*\n\n"
        f"{new_content}"
    )

    context_path.write_text(content + update_block, encoding="utf-8")
    print(f"[obsidian_bridge] Updated context section: {section}")
    return f"Context template updated: {section}"


def list_vault_notes() -> dict:
    """
    Returns a dictionary of all notes in the vault organized by folder.
    Claude Code uses this to understand what already exists before
    creating new notes.
    """
    vault_map = {}
    for folder_name, folder_path in PATHS.items():
        if folder_name in ("moc", "context"):
            continue
        if folder_path.is_dir():
            notes = [f.stem for f in folder_path.glob("*.md")]
            vault_map[folder_name] = notes
            print(f"[obsidian_bridge] {folder_name}/: {notes}")
    return vault_map


def log_code_change(files_changed: list, summary: str) -> str:
    """
    Appends a timestamped change log entry to Code/app_py_Architecture.md.
    Claude Code calls this whenever it modifies app.py or validation_module.py.

    Parameters
    ----------
    files_changed : list  List of filenames that were modified
    summary       : str   Plain English description of what changed and why

    Returns
    -------
    str  Confirmation message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    files_str = ", ".join(files_changed)

    change_entry = (
        f"## Change — {timestamp}\n"
        f"- **Files:** {files_str}\n"
        f"- **Summary:** {summary}\n"
    )

    return append_to_note("Code/app_py_Architecture.md", change_entry)


# ============================================================
# CLI INTERFACE
# Claude Code calls this script via subprocess.
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Obsidian vault bridge for PF-WRP Claude Code sessions"
    )
    parser.add_argument("--action", required=True, choices=[
        "read_context",
        "read_moc",
        "log_session",
        "append_note",
        "new_note",
        "update_task",
        "update_context",
        "list_notes",
        "log_code_change",
    ])
    parser.add_argument("--date",    default=None)
    parser.add_argument("--file",    default=None, help="Relative path from vault root")
    parser.add_argument("--folder",  default=None, help="Vault subfolder name")
    parser.add_argument("--title",   default=None, help="Note title (no .md)")
    parser.add_argument("--content", default="", help="Markdown content to write")
    parser.add_argument("--task",    default=None, help="Task text to search for")
    parser.add_argument("--done",    action="store_true", help="Mark task as done")
    parser.add_argument("--section", default=None, help="Section to update in context")
    parser.add_argument("--files",   default=None, help="Comma-separated changed files")
    parser.add_argument("--summary", default=None, help="Code change summary")

    args = parser.parse_args()

    # Verify vault exists before doing anything
    if not VAULT_PATH.exists():
        print(f"ERROR: Vault not found at {VAULT_PATH}")
        print("Update VAULT_PATH in obsidian_bridge.py to match your vault location.")
        sys.exit(1)

    if args.action == "read_context":
        read_context()

    elif args.action == "read_moc":
        read_moc()

    elif args.action == "log_session":
        log_session(date=args.date or datetime.now().strftime("%Y-%m-%d"))

    elif args.action == "append_note":
        if not args.file or not args.content:
            print("ERROR: --file and --content required for append_note")
            sys.exit(1)
        print(append_to_note(args.file, args.content))

    elif args.action == "new_note":
        if not args.folder or not args.title:
            print("ERROR: --folder and --title required for new_note")
            sys.exit(1)
        print(create_new_note(args.folder, args.title, args.content))

    elif args.action == "update_task":
        if not args.task:
            print("ERROR: --task required for update_task")
            sys.exit(1)
        print(update_task_in_moc(args.task, mark_done=args.done))

    elif args.action == "update_context":
        if not args.section or not args.content:
            print("ERROR: --section and --content required for update_context")
            sys.exit(1)
        print(update_context_template(args.section, args.content))

    elif args.action == "list_notes":
        list_vault_notes()

    elif args.action == "log_code_change":
        if not args.files or not args.summary:
            print("ERROR: --files and --summary required for log_code_change")
            sys.exit(1)
        files_list = [f.strip() for f in args.files.split(",")]
        print(log_code_change(files_list, args.summary))


if __name__ == "__main__":
    main()
