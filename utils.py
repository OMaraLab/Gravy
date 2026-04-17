import csv
import os
import sys

from chemistry_data_structure import parsing


def suppress_output(func):
    def wrapper(*args, **kwargs):
        # Save the original file descriptors
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)

        try:
            # Open /dev/null and redirect stdout and stderr
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)  # Redirect stdout to /dev/null
            os.dup2(devnull, 2)  # Redirect stderr to /dev/null

            return func(*args, **kwargs)

        finally:
            # Restore the original file descriptors
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)

            # Close the duplicated file descriptors
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
            os.close(devnull)

    return wrapper


def progress_bar(iterable, prefix="", total=None):
    total = total or len(iterable)
    for i, item in enumerate(iterable, start=1):
        percent = i / total
        bar_length = 40
        bar = "=" * int(percent * bar_length)
        sys.stdout.write(f"\r{prefix}[{bar:<40}] {i}/{total}")
        sys.stdout.flush()
        yield item
    print()


def load_charges(csv_path: str = "netcharges.csv") -> dict:
    """
    Load net charges into a dict of [molid, charge] lists. The CSV file should contain two columns, molID and net charge
    """

    print(f"Please enter the delimiter used in the CSV file:")
    delimiter = input()
    with open(csv_path, newline="") as csvfile:
        data = csv.reader(csvfile, delimiter=delimiter)
        net_charges = {}
        for x in data:
            net_charges[x[0]] = int(x[1])
        return net_charges
