"""Generate docs/research/phase_3_3/phase_3_3_claims_ledger.md from phase_3_3_claims_ledger.json.

Ensures that the JSON and Markdown versions of the claims ledger cannot diverge.
"""

from __future__ import annotations

import json
from pathlib import Path

LEDGER_DIR = Path("docs/research/phase_3_3")
JSON_PATH = LEDGER_DIR / "phase_3_3_claims_ledger.json"
MD_PATH = LEDGER_DIR / "phase_3_3_claims_ledger.md"


def generate_markdown() -> None:
    claims = json.loads(JSON_PATH.read_text())

    lines = [
        "# AgentXRay — Phase 3.3: Master Claims and Provenance Ledger",
        "",
        "**Document Reference:** `docs/research/phase_3_3/phase_3_3_claims_ledger.md`  ",
        "**Machine-Readable Companion:** [`phase_3_3_claims_ledger.json`](phase_3_3_claims_ledger.json)  ",
        "**Date:** 2026-10-17  ",
        "**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  ",
        "**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  ",
        f"**Total Registered Claims:** {len(claims)}",
        "",
        "---",
        "",
        "## 1. Overview and Ledger Integrity Guarantee",
        "",
        "This document is the authoritative, human-readable rendering of the machine-readable claims ledger `phase_3_3_claims_ledger.json`.",
        "Every single major empirical finding, protocol parameter, statistical estimand, implementation correction, and cryptographic checksum",
        "in Project AgentXRay is registered here with its independent reproduction method and verification status.",
        "",
        "### Consistency Status Key:",
        "- **`VERIFIED`**: Claim independently reproduced bit-for-bit from raw execution artifacts or source code.",
        "- **`CORRECTED`**: Historical documentation erratum identified, reconciled with raw artifacts, and explicitly corrected.",
        "- **`REPORTED_NOT_REPRODUCED`**: Claim stated in prose but cannot be derived from primary records.",
        "- **`UNRESOLVED`**: Data is insufficient to verify claim offline without model re-execution.",
        "",
        "---",
        "",
        "## 2. Master Claims Table",
        "",
        "| Claim ID | Category | Status | Claim Summary | Reproduced Value | Authoritative Source |",
        "| :--- | :--- | :---: | :--- | :--- | :--- |",
    ]

    for c in claims:
        cid = f"`{c['claim_id']}`"
        cat = c["claim_category"]
        stat = f"**{c['consistency_status']}**"
        text = c["claim_text"]
        val = str(c["reproduced_value"]).replace("|", "\\|")
        if len(val) > 45:
            val = val[:42] + "..."
        val_code = f"`{val}`"
        src = f"`{c['authoritative_source_artifact']}`"
        lines.append(f"| {cid} | {cat} | {stat} | {text} | {val_code} | {src} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Claim Specifications",
        "",
    ])

    for c in claims:
        lines.extend([
            f"### {c['claim_id']}",
            f"- **Statement:** {c['claim_text']}",
            f"- **Category:** `{c['claim_category']}` | **Status:** `{c['consistency_status']}`",
            f"- **Source Artifact:** `{c['authoritative_source_artifact']}`",
            f"- **Source Fields / Record Identifiers:** `{c['source_fields_or_identifiers']}`",
            f"- **Reproduction Method:** {c['reproduction_method']}",
            f"- **Original Reported Value:** `{c['original_reported_value']}`",
            f"- **Independently Reproduced Value:** `{c['reproduced_value']}`",
            f"- **Caveats & Notes:** {c['caveats']}",
            "",
        ])

    MD_PATH.write_text("\n".join(lines))
    print(f"Generated {MD_PATH} from {JSON_PATH} successfully ({len(claims)} claims).")


if __name__ == "__main__":
    generate_markdown()
