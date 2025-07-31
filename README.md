# SSH Log Analyzer using K-Clique Clustering

This project provides a tool to analyze SSH authentication logs and detect potential brute-force attacks by identifying clusters of suspicious IP addresses. It uses the k-clique percolation algorithm to find groups of IP addresses that have attempted to log in with the same usernames.

## Features

- **SSH Log Parsing**: Extracts failed login attempts from standard SSH log files.
- **Graph-Based Clustering**: Builds a graph of IP addresses and uses k-clique percolation to find communities of related IPs.
- **Visualization**: Generates a graph visualization of the IP network and the detected clusters, saved as a PNG image.
- **Command-Line Interface**: Easy-to-use CLI to specify the log file, k-value, and output file.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the analysis from the command line. You need to provide the path to your SSH log file.

### Basic Usage

To analyze a log file named `ssh_logs.txt` with the default k-value of 3:
```bash
python3 ssh_log_analyzer/main.py ssh_logs.txt
```
This will print the found clusters to the console and generate a `clusters.png` file in the root directory.

### Options

-   `-k <value>`: Specify the value of k for the k-clique algorithm. This determines the minimum size of the cliques to be considered.
-   `-o <filename>` or `--output <filename>`: Specify the name of the output image file.

### Example with Options

To run the analysis with `k=4` and save the output to `my_analysis.png`:
```bash
python3 ssh_log_analyzer/main.py ssh_logs.txt -k 4 -o my_analysis.png
```
