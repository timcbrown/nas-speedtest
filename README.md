# NAS Speed Test

A Python script to test Network Attached Storage (NAS) performance across different computers, with special attention to the differences between bulk file transfers and operations involving many small files.

## Features

- Measures read/write speeds for large files and many small files
- Tests filesystem latency
- Works on Windows, macOS, and Linux
- No external dependencies (standard library only)
- Compares performance ratios between large and small file operations

## Usage
python3 nas-speedtest.py [options] nas_path

### Options

- `--large-size SIZE`: Size of large test file in MB (default: 100)
- `--small-count COUNT`: Number of small files to test (default: 500)
- `--iterations N`: Number of test iterations (default: 3)
- `--timeout SECONDS`: Timeout in seconds (default: 30)
- `--output FILE`: Save results to specified file
- `--verbose`: Enable verbose output
- `--skip-tests TESTS`: Comma-separated list of tests to skip
