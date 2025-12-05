"""
Test Runner Script for Sustainable AI Project.
Runs all tests and generates reports.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --integration # Run integration tests only
    python run_tests.py --e2e        # Run E2E tests only
    python run_tests.py --coverage   # Run with coverage
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
TESTS_DIR = PROJECT_ROOT / "tests"


def run_tests(test_type: str = "all", coverage: bool = False, verbose: bool = True):
    """Run tests based on type."""
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html:reports/coverage",
            "--cov-report=term-missing"
        ])
    
    # Select test type
    if test_type == "unit":
        cmd.extend([
            str(TESTS_DIR / "test_config.py"),
            str(TESTS_DIR / "test_nlp.py"),
            str(TESTS_DIR / "test_predictor.py"),
            str(TESTS_DIR / "test_detector.py"),
            str(TESTS_DIR / "test_optimizer.py"),
            str(TESTS_DIR / "test_database.py"),
            str(TESTS_DIR / "test_gui.py"),
        ])
    elif test_type == "integration":
        cmd.extend(["-m", "integration", str(TESTS_DIR)])
    elif test_type == "e2e":
        cmd.extend(["-m", "e2e", str(TESTS_DIR / "test_e2e.py")])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow", str(TESTS_DIR)])
    else:
        # All tests
        cmd.append(str(TESTS_DIR))
    
    # Add HTML report
    cmd.extend([
        "--html=reports/test_report.html",
        "--self-contained-html"
    ])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run Sustainable AI tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run E2E tests")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    # Create reports directory
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Determine test type
    if args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    elif args.e2e:
        test_type = "e2e"
    elif args.fast:
        test_type = "fast"
    else:
        test_type = "all"
    
    # Run tests
    result = run_tests(
        test_type=test_type,
        coverage=args.coverage,
        verbose=not args.quiet
    )
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
