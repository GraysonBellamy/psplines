#!/usr/bin/env python3
"""
Test script to verify documentation builds correctly.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and check for errors."""
    print(f"\n🔧 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        return False
    else:
        print(f"✅ Success: {description}")
        return True


def main():
    """Main test function."""
    print("📚 Testing PSplines Documentation")
    print("=" * 50)

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    tests = [
        ("mkdocs --version", "Check MkDocs installation"),
        ("mkdocs build --strict", "Build documentation"),
        (
            "python -c \"import psplines; print('PSplines import successful')\"",
            "Test PSplines import",
        ),
    ]

    # Additional checks
    print("\n📋 Checking required files...")
    required_files = [
        "mkdocs.yml",
        "docs/index.md",
        "docs/api/core.md",
        "docs/user-guide/getting-started.md",
        "docs/tutorials/basic-usage.md",
        "docs/examples/gallery.md",
    ]

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            return False

    # Run tests
    all_passed = True
    for cmd, description in tests:
        if not run_command(cmd, description):
            all_passed = False

    # Check if site was built
    if Path("site").exists() and Path("site/index.html").exists():
        print("\n✅ Site built successfully")
        print(f"📁 Site directory: {Path('site').absolute()}")

        # Count pages
        html_files = list(Path("site").rglob("*.html"))
        print(f"📄 Generated {len(html_files)} HTML pages")

    else:
        print("\n❌ Site build failed - no site directory found")
        all_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All documentation tests passed!")
        print("💡 To serve locally, run: mkdocs serve")
        return 0
    else:
        print("❌ Some documentation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
