"""
Command-line interface for Delta.

Provides commands for compiling and running Delta programs.
"""

import argparse
import sys
from pathlib import Path

import delta


def main():
    """Main entry point for the delta CLI."""
    parser = argparse.ArgumentParser(
        prog="delta",
        description="Delta: A differentiable, constraint-oriented programming language",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"delta {delta.__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile a Delta program")
    compile_parser.add_argument("file", type=str, help="Delta source file (.delta)")
    compile_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file (default: <input>.py)"
    )
    compile_parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Type-check only, don't generate code"
    )
    compile_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Compile and run a Delta program")
    run_parser.add_argument("file", type=str, help="Delta source file (.delta)")
    run_parser.add_argument(
        "--mode",
        choices=["train", "infer", "analyze"],
        default="train",
        help="Execution mode (default: train)"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "compile":
        return do_compile(args)
    elif args.command == "run":
        return do_run(args)


def do_compile(args):
    """Handle the compile command."""
    source_path = Path(args.file)
    
    if not source_path.exists():
        print(f"Error: File not found: {source_path}", file=sys.stderr)
        return 1
    
    if not source_path.suffix == ".delta":
        print(f"Warning: File does not have .delta extension: {source_path}", file=sys.stderr)
    
    try:
        if args.verbose:
            print(f"Compiling {source_path}...")
        
        model = delta.compile(str(source_path))
        
        if args.check:
            print(f"✓ {source_path}: Type check passed")
            return 0
        
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = source_path.with_suffix(".py")
        
        # For now, just confirm compilation succeeded
        print(f"✓ Compiled {source_path}")
        
        if args.verbose:
            print(f"  Parameters: {len(model.params)}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def do_run(args):
    """Handle the run command."""
    source_path = Path(args.file)
    
    if not source_path.exists():
        print(f"Error: File not found: {source_path}", file=sys.stderr)
        return 1
    
    try:
        model = delta.compile(str(source_path))
        
        if args.mode == "train":
            model.train()
        elif args.mode == "infer":
            model.eval()
        
        print(f"Model loaded in {args.mode} mode")
        print(f"Parameters: {len(model.params)}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def compile_cmd():
    """Entry point for deltac command."""
    parser = argparse.ArgumentParser(
        prog="deltac",
        description="Compile a Delta program",
    )
    parser.add_argument("file", type=str, help="Delta source file (.delta)")
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Type-check only"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    sys.exit(do_compile(args))


if __name__ == "__main__":
    sys.exit(main() or 0)
