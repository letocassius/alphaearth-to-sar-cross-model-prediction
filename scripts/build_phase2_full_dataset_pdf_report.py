#!/usr/bin/env python3
"""
Compatibility wrapper for the project reporting pipeline.

The current reporting workflow generates:
- a Phase 2 summary PDF
- a Phase 3 summary PDF
- a cumulative project summary PDF
"""

from build_project_reports import main


if __name__ == "__main__":
    main()
