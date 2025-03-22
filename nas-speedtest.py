#!/usr/bin/env python3
"""
NAS Speed Test - Tests Network Attached Storage (NAS) performance with both
large files and many small files across different platforms.

Usage: python nas_speed_test.py [options] nas_path
"""

import os
import sys
import time
import shutil
import random
import platform
import argparse
import subprocess
import tempfile
import socket
import logging
import signal
import contextlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Callable, Any


# Constants
VALID_TESTS = {'latency', 'fs-latency', 'large-read', 'large-write', 'small-read', 'small-write'}


# Simple timeout context manager
@contextlib.contextmanager
def timeout_context(seconds):
    """Context manager that raises TimeoutError after specified seconds."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    if seconds is None or platform.system() == 'Windows':
        # Windows doesn't support SIGALRM, just yield
        yield
        return

    # Set the timeout handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


# Test result dataclass
@dataclass
class TestResult:
    success: bool = True
    error: str = None
    data: Dict = field(default_factory=dict)
    
    def __bool__(self):
        return self.success


# Command-line argument parsing
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test NAS connection speed with various file operations')
    parser.add_argument('nas_path', help='Path to NAS directory for testing')
    parser.add_argument('--large-size', type=int, default=100,
                        help='Size of large test file in MB (default: 100)')
    parser.add_argument('--small-count', type=int, default=500,
                        help='Number of small files to test (default: 500)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of test iterations (default: 3)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout in seconds for network operations (default: 30)')
    parser.add_argument('--output', type=str, help='Save results to specified file')
    parser.add_argument('--skip-tests', type=str, 
                        help=f'Comma-separated list of tests to skip ({",".join(VALID_TESTS)})')
    parser.add_argument('--keep-outliers', action='store_true',
                        help='Do not remove outliers when calculating averages')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel file generation for small files test')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--dry-run', action='store_true', help='Simulate actions without performing tests')
    
    args = parser.parse_args()
    
    # Validate arguments
    for arg_name, min_val in [('large-size', 1), ('small-count', 1), ('iterations', 1), ('timeout', 1)]:
        if getattr(args, arg_name.replace('-', '_')) < min_val:
            parser.error(f"{arg_name} must be at least {min_val}")
    
    if args.skip_tests:
        skip_tests = set(args.skip_tests.split(','))
        invalid_tests = skip_tests - VALID_TESTS
        if invalid_tests:
            parser.error(f"Invalid test names: {', '.join(invalid_tests)}. Valid: {', '.join(VALID_TESTS)}")
    
    return args


# Set up logging
def configure_logging(verbose, log_file=None):
    """Configure logging with optional file output."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Root logger configuration
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    
    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logger.addHandler(file_handler)
        except (PermissionError, IOError) as e:
            logger.error(f"Could not create log file: {e}")
    
    return logger


# System information gathering
def get_system_info():
    """Get system information for reporting."""
    info = {
        'hostname': socket.gethostname(),
        'os': platform.system(),
        'os_version': platform.version(),
        'python_version': platform.python_version(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'network': f'Unknown ({platform.system()})'
    }
    
    # Platform-specific network detection
    commands = {
        'Windows': ('ipconfig /all', lambda out: next((l.strip() for l in out.split('\n') 
                    if 'speed' in l.lower() and ('gbps' in l.lower() or 'mbps' in l.lower())), None)),
        'Linux': (lambda: {'cmd': 'ip route get 1.1.1.1', 
                          'parser': lambda out: next((l.split('dev')[1].strip().split()[0] for l in out.split('\n') if 'dev' in l), None),
                          'next': lambda iface: {'cmd': f'ethtool {iface}', 
                                               'parser': lambda out: f"{iface}: {next((l.strip() for l in out.split('\n') if 'speed' in l.lower()), 'Unknown speed')}"}}),
        'Darwin': (lambda: {'cmd': 'networksetup -listallhardwareports', 
                           'parser': lambda out: next((f"{curr_port} ({dev}): Active" for curr_port, dev in 
                                                     ((l.split(':')[1].strip(), next_l.split(':')[1].strip()) 
                                                      for l, next_l in zip(out.split('\n'), out.split('\n')[1:]) 
                                                      if 'Hardware Port' in l and 'Device' in next_l) 
                                                     if subprocess.run(['ifconfig', dev], capture_output=True, text=True).stdout.find('status: active') != -1), None)})
    }
    
    try:
        cmd_info = commands.get(info['os'])
        if not cmd_info:
            return info
            
        if callable(cmd_info):
            cmd_info = cmd_info()
            
        if isinstance(cmd_info, tuple):
            cmd, parser = cmd_info
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                network = parser(result.stdout)
                if network:
                    info['network'] = network
        elif isinstance(cmd_info, dict):
            result = subprocess.run(cmd_info['cmd'].split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parsed = cmd_info['parser'](result.stdout)
                if parsed and 'next' in cmd_info:
                    next_cmd = cmd_info['next'](parsed)
                    next_result = subprocess.run(next_cmd['cmd'].split(), capture_output=True, text=True, timeout=5)
                    if next_result.returncode == 0:
                        network = next_cmd['parser'](next_result.stdout)
                        if network:
                            info['network'] = network
                elif parsed:
                    info['network'] = parsed
    except Exception as e:
        logging.debug(f"Network detection error: {e}")
    
    return info


# Utility functions
def generate_random_file(path, size_mb):
    """Generate a file of specified size with random content."""
    logging.info(f"Generating {size_mb}MB random file at {path}")
    
    size_bytes = size_mb * 1024 * 1024
    chunk_size = min(4 * 1024 * 1024, size_bytes)  # 4MB chunks
    
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        remaining = size_bytes
        while remaining > 0:
            current_chunk = min(chunk_size, remaining)
            f.write(os.urandom(current_chunk))
            remaining -= current_chunk
            
            # Progress for large files
            if size_mb > 10 and (size_bytes - remaining) % (10 * 1024 * 1024) == 0:
                logging.debug(f"  {((size_bytes - remaining) / size_bytes * 100):.1f}% complete")
        
        f.flush()
        os.fsync(f.fileno())
    
    return path


def generate_file_structure(base_path, file_count, min_size_kb=1, max_size_kb=100, use_parallel=True):
    """Generate a directory structure with many small files."""
    logging.info(f"Generating {file_count} files structure at {base_path}")
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    # Create directory tree
    depth = min(3, file_count // 50)  # Reasonable depth based on file count
    dirs = [base_path]
    
    for level in range(1, depth + 1):
        dirs.extend([Path(parent) / f"dir_{level}_{i}" for parent in dirs 
                    for i in range(min(5, file_count // 20))])
    
    # Create all directories
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    
    # Prepare file creation tasks
    tasks = []
    total_dirs = len(dirs)
    total_size = 0
    
    for i in range(file_count):
        dir_path = dirs[i % total_dirs]  # Distribute evenly
        file_path = dir_path / f"file_{i}.dat"
        size_kb = random.randint(min_size_kb, max_size_kb)
        tasks.append((file_path, size_kb))
    
    # Create files (parallel or sequential)
    if use_parallel and file_count > 50:
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
            for i, size in enumerate(executor.map(lambda t: _create_small_file(*t), tasks)):
                total_size += size
                if (i + 1) % 50 == 0 or (i + 1) == file_count:
                    logging.debug(f"  Created {i+1}/{file_count} files")
    else:
        for i, (file_path, size_kb) in enumerate(tasks):
            total_size += _create_small_file(file_path, size_kb)
            if (i + 1) % 50 == 0 or (i + 1) == file_count:
                logging.debug(f"  Created {i+1}/{file_count} files")
    
    return {'path': base_path, 'file_count': file_count, 'total_size': total_size}


def _create_small_file(path, size_kb):
    """Create a small file with random content."""
    content = os.urandom(size_kb * 1024)
    with open(path, 'wb') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    return len(content)


def extract_nas_host(nas_path):
    """Extract hostname from NAS path for ping tests."""
    path_str = str(nas_path)
    
    # Different path patterns
    if path_str.startswith('\\\\'):  # Windows UNC
        return path_str.split('\\')[2]
    elif path_str.startswith('smb://'):  # SMB URL
        return path_str.split('/')[2]
    elif ':/' in path_str and not path_str.startswith('/'):  # NFS
        return path_str.split(':')[0]
    
    # Try to find IP pattern
    import re
    ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', path_str)
    if ip_match:
        return ip_match.group(0)
    
    # Default to local hostname for local paths
    return socket.gethostname() if Path(path_str).exists() else "unknown-host"


def cleanup(paths):
    """Clean up temporary files and directories."""
    for path in paths:
        try:
            path = Path(path)
            if not path.exists():
                continue
                
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
            logging.debug(f"Cleaned up: {path}")
        except Exception as e:
            logging.debug(f"Cleanup error for {path}: {e}")


def calculate_statistics(values, remove_outliers=True):
    """Calculate statistics from a list of values."""
    if not values:
        return {'mean': 0, 'min': 0, 'max': 0, 'count': 0}
    
    if remove_outliers and len(values) >= 3:
        # Remove values more than 2 standard deviations from mean
        mean = sum(values) / len(values)
        stdev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        values = [v for v in values if abs(v - mean) <= 2 * stdev]
    
    return {
        'mean': sum(values) / len(values) if values else 0,
        'min': min(values) if values else 0,
        'max': max(values) if values else 0,
        'count': len(values)
    }


# Test functions
def test_latency(nas_host, timeout=10):
    """Test network latency to NAS host using ping."""
    logging.info(f"Testing latency to {nas_host}...")
    
    # Platform-specific ping command
    command = ['ping', '-n', '5', nas_host] if platform.system() == 'Windows' else ['ping', '-c', '5', nas_host]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
        output = result.stdout
        
        # Parse based on platform
        if platform.system() == 'Windows':
            avg_line = next((l for l in output.split('\n') if 'Average' in l), None)
            if not avg_line:
                return TestResult(False, "Failed to parse Windows ping output")
                
            avg_ms = float(avg_line.split('=')[-1].strip().replace('ms', '').strip())
            min_max_line = next((l for l in output.split('\n') if 'Minimum' in l), None)
            
            if not min_max_line:
                return TestResult(success=False, error="Failed to parse Windows ping output")
                
            parts = min_max_line.split('=')[-1].strip().split(', ')
            min_ms = float(parts[0].replace('ms', '').strip())
            max_ms = float(parts[1].replace('ms', '').strip())
            
            return TestResult(data={'avg': avg_ms, 'min': min_ms, 'max': max_ms})
        else:
            rtt_line = next((l for l in output.split('\n') 
                         if 'rtt min/avg/max' in l or 'round-trip min/avg/max' in l), None)
                         
            if not rtt_line:
                return TestResult(success=False, error="Failed to parse ping output")
                
            stats = rtt_line.split('=')[1].strip().split('/')
            return TestResult(data={
                'min': float(stats[0]),
                'avg': float(stats[1]),
                'max': float(stats[2])
            })
    except subprocess.SubprocessError as e:
        return TestResult(success=False, error=f"Ping failed: {e}")


def test_filesystem_latency(path, iterations=10, timeout=30):
    """Test filesystem-specific latency using small file operations."""
    logging.info(f"Testing filesystem latency on {path}...")
    
    test_dir = Path(path) / "latency_test"
    try:
        with timeout_context(timeout):
            test_dir.mkdir(exist_ok=True)
            
            # Warm-up operations
            for _ in range(2):
                warm_up_file = test_dir / "warmup.tmp"
                with open(warm_up_file, 'wb') as f:
                    f.write(b'x')
                    f.flush()
                    os.fsync(f.fileno())
                warm_up_file.unlink()
            
            # Test operations
            times = {'creation': [], 'read': [], 'delete': []}
            
            for i in range(iterations):
                test_file = test_dir / f"latency_test_{i}.tmp"
                
                # Creation time
                start = time.time()
                with open(test_file, 'wb') as f:
                    f.write(b'x')
                    f.flush()
                    os.fsync(f.fileno())
                times['creation'].append((time.time() - start) * 1000)
                
                # Read time
                start = time.time()
                with open(test_file, 'rb') as f:
                    f.read()
                times['read'].append((time.time() - start) * 1000)
                
                # Delete time
                start = time.time()
                test_file.unlink()
                times['delete'].append((time.time() - start) * 1000)
            
            # Calculate statistics for each operation
            stats = {}
            for op, values in times.items():
                stats[op] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
            
            return TestResult(data=stats)
    except Exception as e:
        return TestResult(success=False, error=str(e))
    finally:
        # Clean up
        try:
            if test_dir.exists():
                shutil.rmtree(test_dir)
        except Exception as e:
            logging.warning(f"Could not clean up latency test directory: {e}")


def test_file_transfer(src_path, dest_dir, direction='read', iterations=3, timeout=30):
    """Test file transfer speed (read or write).
    
    Args:
        src_path: Source file or directory
        dest_dir: Destination directory
        direction: 'read' (NAS to local) or 'write' (local to NAS)
        iterations: Number of test iterations for large files
        timeout: Operation timeout in seconds
    """
    src_path = Path(src_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if we're testing a large file or small files
    is_large_file = src_path.is_file()
    operation = f"{'large' if is_large_file else 'small'}-{direction}"
    
    # For large files, get size and prepare destination path
    if is_large_file:
        file_size = src_path.stat().st_size
        dest_path = dest_dir / src_path.name
        logging.info(f"Testing {operation} ({file_size / (1024*1024):.1f} MB)")
    else:
        # For small files, count files and total size
        file_count = 0
        total_size = 0
        for root, _, files in os.walk(src_path):
            for filename in files:
                file_path = Path(root) / filename
                total_size += file_path.stat().st_size
                file_count += 1
        
        dest_path = dest_dir / src_path.name
        logging.info(f"Testing {operation} ({file_count} files, {total_size / (1024*1024):.1f} MB total)")
    
    try:
        with timeout_context(timeout):
            # For large files, do multiple iterations
            if is_large_file:
                speeds = []
                
                for i in range(iterations):
                    # Clean up any existing file
                    if dest_path.exists():
                        dest_path.unlink()
                    
                    # Perform transfer and measure time
                    start_time = time.time()
                    shutil.copy2(src_path, dest_path)
                    
                    # Ensure data is flushed
                    with open(dest_path, 'rb') as f:
                        os.fsync(f.fileno())
                    
                    duration = time.time() - start_time
                    speed_mbps = (file_size / duration) / (1024 * 1024)
                    speeds.append(speed_mbps)
                    
                    logging.info(f"  Iteration {i+1}: {speed_mbps:.2f} MB/s")
                    
                    # Clean up after each iteration
                    if dest_path.exists():
                        dest_path.unlink()
                
                # Calculate statistics
                stats = calculate_statistics(speeds)
                return TestResult(data={
                    'file_size': file_size,
                    'avg_speed': stats['mean'],
                    'min_speed': stats['min'],
                    'max_speed': stats['max'],
                    'iterations': len(speeds)
                })
            else:
                # For small files, single iteration
                start_time = time.time()
                
                # Copy directory structure
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
                
                # Ensure all files are flushed
                for root, _, files in os.walk(dest_path):
                    for filename in files:
                        file_path = Path(root) / filename
                        with open(file_path, 'rb') as f:
                            os.fsync(f.fileno())
                
                # Calculate metrics
                duration = time.time() - start_time
                speed_mbps = (total_size / duration) / (1024 * 1024)
                files_per_sec = file_count / duration
                
                logging.info(f"  Speed: {speed_mbps:.2f} MB/s ({files_per_sec:.1f} files/sec)")
                
                return TestResult(data={
                    'file_count': file_count,
                    'total_size': total_size,
                    'duration': duration,
                    'speed_mbps': speed_mbps,
                    'files_per_sec': files_per_sec
                })
    except Exception as e:
        operation_name = "read" if direction == "read" else "write"
        file_type = "large file" if is_large_file else "small files"
        return TestResult(success=False, error=f"Error during {file_type} {operation_name}: {e}")


# Results output
def format_test_results(results, sys_info):
    """Format test results into a readable string."""
    lines = [
        "NAS Speed Test Results",
        "---------------------",
        f"Date: {sys_info['timestamp']}",
        f"Computer: {sys_info['hostname']} ({sys_info['os']})",
        f"NAS Path: {results['nas_path']}",
        f"Network Interface: {sys_info['network']}",
        ""
    ]
    
    # Network latency
    if 'latency' in results:
        lines.append("Latency Test:")
        latency = results['latency']
        if latency.success:
            lines.append(f"  Average: {latency.data['avg']:.1f} ms")
            lines.append(f"  Min: {latency.data['min']:.1f} ms")
            lines.append(f"  Max: {latency.data['max']:.1f} ms")
        else:
            lines.append(f"  Error: {latency.error}")
        lines.append("")
    
    # Filesystem latency
    if 'fs_latency' in results:
        lines.append("Filesystem Latency Test:")
        fs_latency = results['fs_latency']
        if fs_latency.success:
            lines.append(f"  Creation: {fs_latency.data['creation']['avg']:.1f} ms")
            lines.append(f"  Read: {fs_latency.data['read']['avg']:.1f} ms")
            lines.append(f"  Delete: {fs_latency.data['delete']['avg']:.1f} ms")
        else:
            lines.append(f"  Error: {fs_latency.error}")
        lines.append("")
    
    # Large file test
    if 'large_read' in results or 'large_write' in results:
        large_read = results.get('large_read')
        large_write = results.get('large_write')
        
        if (large_read and large_read.success) or (large_write and large_write.success):
            size_mb = large_read.data['file_size'] / (1024*1024) if large_read and large_read.success else \
                      large_write.data['file_size'] / (1024*1024)
            
            lines.append(f"Large File Test ({size_mb:.1f} MB):")
            
            if large_read:
                if large_read.success:
                    lines.append(f"  Read: {large_read.data['avg_speed']:.1f} MB/s")
                else:
                    lines.append(f"  Read Error: {large_read.error}")
            
            if large_write:
                if large_write.success:
                    lines.append(f"  Write: {large_write.data['avg_speed']:.1f} MB/s")
                else:
                    lines.append(f"  Write Error: {large_write.error}")
            
            lines.append("")
    
    # Small files test
    if 'small_read' in results or 'small_write' in results:
        small_read = results.get('small_read')
        small_write = results.get('small_write')
        
        if (small_read and small_read.success) or (small_write and small_write.success):
            # Get file count and size from available test
            if small_read and small_read.success:
                file_count = small_read.data['file_count']
                total_size = small_read.data['total_size'] / (1024*1024)
            else:
                file_count = small_write.data['file_count']
                total_size = small_write.data['total_size'] / (1024*1024)
            
            lines.append(f"Small Files Test ({file_count} files, {total_size:.1f} MB total):")
            
            if small_read:
                if small_read.success:
                    lines.append(f"  Read: {small_read.data['speed_mbps']:.1f} MB/s ({small_read.data['files_per_sec']:.0f} files/sec)")
                else:
                    lines.append(f"  Read Error: {small_read.error}")
            
            if small_write:
                if small_write.success:
                    lines.append(f"  Write: {small_write.data['speed_mbps']:.1f} MB/s ({small_write.data['files_per_sec']:.0f} files/sec)")
                else:
                    lines.append(f"  Write Error: {small_write.error}")
            
            lines.append("")
    
    # Summary ratios
    lines.append("Summary:")
    
    # Calculate large vs small ratios
    if ('large_read' in results and 'small_read' in results and 
        results['large_read'].success and results['small_read'].success):
        read_ratio = results['large_read'].data['avg_speed'] / results['small_read'].data['speed_mbps']
        lines.append(f"  Large vs Small Files Read Ratio: {read_ratio:.2f}")
    
    if ('large_write' in results and 'small_write' in results and
        results['large_write'].success and results['small_write'].success):
        write_ratio = results['large_write'].data['avg_speed'] / results['small_write'].data['speed_mbps']
        lines.append(f"  Large vs Small Files Write Ratio: {write_ratio:.2f}")
    
    return "\n".join(lines)


def save_results(formatted_results, filename):
    """Save formatted results to a file."""
    try:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text(formatted_results)
        logging.info(f"Results saved to {filename}")
        return True
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        return False


# Main function
def main():
    """Main function to orchestrate NAS speed tests."""
    args = parse_arguments()
    logger = configure_logging(args.verbose, args.output + ".log" if args.output else None)
    
    # Determine tests to skip
    skip_tests = set(args.skip_tests.split(',') if args.skip_tests else [])
    
    # Validate NAS path
    nas_path = Path(args.nas_path)
    if not args.dry_run and not nas_path.exists():
        logging.error(f"NAS path does not exist: {nas_path}")
        return 1
    
    # Log test parameters
    logging.info("NAS Speed Test")
    logging.info("-------------")
    logging.info(f"NAS Path: {nas_path}")
    logging.info(f"Tests: {', '.join(sorted(t for t in VALID_TESTS if t not in skip_tests))}")
    logging.info("")
    
    # Get system information
    sys_info = get_system_info()
    logging.info(f"System: {sys_info['hostname']} ({sys_info['os']})")
    logging.info(f"Network: {sys_info['network']}")
    logging.info("")
    
    # Initialize results and temp directories
    results = {'nas_path': str(nas_path)}
    temp_dirs = []
    
    try:
        # Create temporary directories
        local_temp = Path(tempfile.mkdtemp(prefix="nas_speed_test_"))
        temp_dirs.append(local_temp)
        
        nas_temp = nas_path / f"nas_speed_test_{int(time.time())}"
        if not args.dry_run:
            nas_temp.mkdir(exist_ok=True)
            temp_dirs.append(nas_temp)
        
        # Extract NAS host for latency test
        nas_host = extract_nas_host(nas_path)
        
        # Run tests based on configuration
        if 'latency' not in skip_tests and not args.dry_run:
            results['latency'] = test_latency(nas_host, timeout=args.timeout)
        
        if 'fs-latency' not in skip_tests and not args.dry_run:
            results['fs_latency'] = test_filesystem_latency(
                nas_temp, iterations=args.iterations, timeout=args.timeout
            )
        
        # Large file test preparation
        large_file = ""
        if ('large-read' not in skip_tests or 'large-write' not in skip_tests) and not args.dry_run:
            large_file = local_temp / "large_test_file.dat"
            generate_random_file(large_file, args.large_size)
        
        # Large file tests
        if 'large-write' not in skip_tests and large_file and not args.dry_run:
            nas_write_dir = nas_temp / "large_write"
            results['large_write'] = test_file_transfer(
                large_file, nas_write_dir, 'write',
                iterations=args.iterations, timeout=args.timeout * 2
            )
        
        if 'large-read' not in skip_tests and large_file and not args.dry_run:
            # Copy file to NAS first for reading
            nas_large_file = nas_temp / large_file.name
            try:
                shutil.copy2(large_file, nas_large_file)
                local_read_dir = local_temp / "large_read"
                
                results['large_read'] = test_file_transfer(
                    nas_large_file, local_read_dir, 'read',
                    iterations=args.iterations, timeout=args.timeout * 2
                )
            except Exception as e:
                results['large_read'] = TestResult(success=False, error=str(e))
        
        # Small files test preparation
        small_files_dir = ""
        if ('small-read' not in skip_tests or 'small-write' not in skip_tests) and not args.dry_run:
            small_files_dir = local_temp / "small_files"
            generate_file_structure(
                small_files_dir, args.small_count, use_parallel=not args.no_parallel
            )
        
        # Small files tests
        if 'small-write' not in skip_tests and small_files_dir and not args.dry_run:
            nas_small_dir = nas_temp / "small_write"
            results['small_write'] = test_file_transfer(
                small_files_dir, nas_small_dir, 'write', timeout=args.timeout * 3
            )
        
        if 'small-read' not in skip_tests and small_files_dir and not args.dry_run:
            # Copy to NAS first for reading
            nas_small_files = nas_temp / small_files_dir.name
            try:
                shutil.copytree(small_files_dir, nas_small_files)
                local_read_dir = local_temp / "small_read"
                
                results['small_read'] = test_file_transfer(
                    nas_small_files, local_read_dir, 'read', timeout=args.timeout * 3
                )
            except Exception as e:
                results['small_read'] = TestResult(success=False, error=str(e))
        
        # Format and display results
        formatted_results = format_test_results(results, sys_info)
        logging.info("\n" + formatted_results)
        
        # Save results to file if requested
        if args.output:
            save_results(formatted_results, args.output)
    
    except KeyboardInterrupt:
        logging.info("\nTest interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        # Clean up
        logging.info("Cleaning up...")
        cleanup(temp_dirs)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())