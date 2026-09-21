"""
Event Logger for Phase 0 Governance

This script appends structured events to EVENTS.md following the project governance framework.

Usage:
    python -m src.governance.event_logger --event-id E-XXX-NNN --type "Type" --description "Description" --rationale "Rationale" --impact "Impact" --evidence "Evidence"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path


def append_event(events_path: Path, event_id: str, event_type: str, 
                 description: str, rationale: str, impact: str, evidence: str) -> None:
    """Append a new event entry to EVENTS.md."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    event_entry = f"""
### {event_id}
- **Timestamp**: {timestamp}
- **Type**: {event_type}
- **Description**: {description}
- **Rationale**: {rationale}
- **Impact**: {impact}
- **Evidence**: {evidence}
"""
    
    content = events_path.read_text() if events_path.exists() else "# Events Log\n\nThis document records all significant events, decisions, and changes to the LyoXL digital twin project.\n\n---\n"
    
    # Find the position before the closing note (if present)
    closing_note = "\n---\n\n*This log is append-only. New events should be added at the bottom.*"
    if closing_note in content:
        content = content.replace(closing_note, event_entry + closing_note)
    else:
        # Append at the end
        content = content.rstrip() + event_entry + "\n"
    
    events_path.write_text(content)
    print(f"Event {event_id} logged to {events_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Append structured events to EVENTS.md for Phase 0 governance"
    )
    parser.add_argument("--event-id", required=True, help="Unique event identifier (e.g., E-GOV-001)")
    parser.add_argument("--type", required=True, help="Category of change (e.g., Governance, Physics, QC)")
    parser.add_argument("--description", required=True, help="What happened")
    parser.add_argument("--rationale", required=True, help="Why this change was made")
    parser.add_argument("--impact", required=True, help="Effect on project phases/gates")
    parser.add_argument("--evidence", required=True, help="Files or tests that verify the change")
    parser.add_argument("--events-file", type=Path, default=Path("EVENTS.md"),
                        help="Path to EVENTS.md file (default: EVENTS.md)")
    
    args = parser.parse_args()
    
    append_event(
        events_path=args.events_file,
        event_id=args.event_id,
        event_type=args.type,
        description=args.description,
        rationale=args.rationale,
        impact=args.impact,
        evidence=args.evidence
    )


if __name__ == "__main__":
    main()
