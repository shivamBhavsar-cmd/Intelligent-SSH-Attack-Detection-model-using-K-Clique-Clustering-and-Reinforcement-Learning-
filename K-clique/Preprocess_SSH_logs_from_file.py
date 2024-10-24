import re
import networkx as nx
import numpy as np

# Preprocess SSH logs from file
def preprocess_logs(logfile):
    with open(logfile, 'r') as f:
        logs = f.readlines()
    
    pattern = re.compile(r'Failed password for (\w+) from (\d+\.\d+\.\d+\.\d+)')
    parsed_logs = []
    for log in logs:
        match = pattern.search(log)
        if match:
            username, ip = match.groups()
            parsed_logs.append((username, ip))
    return parsed_logs

parsed_logs = preprocess_logs('ssh_logs.txt')
print(parsed_logs)
# Example output: [('root', '192.168.1.1'), ('root', '192.168.1.2'), ('admin', '192.168.1.1')]