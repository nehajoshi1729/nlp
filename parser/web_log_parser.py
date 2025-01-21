import csv
import re
from collections import Counter, defaultdict
from datetime import datetime

def parse_log_line(line):
    """
    Parse a single log line using a regular expression.
    Returns a dictionary of fields if the line matches the expected format, otherwise None.
    """
    log_pattern = re.compile(r'^(\S+) (\S+) (\S+) \[(.*?)\] \"(\S+) (.*?) (\S+)\" (\d{3}) (\S+)$')
    match = log_pattern.match(line)
    if match:
        return {
            "ip": match.group(1),
            "ident": match.group(2),
            "user": match.group(3),
            "timestamp": match.group(4),
            "method": match.group(5),
            "resource": match.group(6),
            "protocol": match.group(7),
            "status": int(match.group(8)),
            "size": match.group(9) if match.group(9) != "-" else None
        }
    return None

def verify_parsing_function():
    """
    Test the parse_log_line function on sample lines and print results.
    """
    test_lines = [
        "123.45.67.89 - - [12/Dec/2024:06:25:17 +0000] \"GET /index.html HTTP/1.1\" 200 5123",
        "55.66.77.88 - admin [12/Dec/2024:06:25:18 +0000] \"POST /login HTTP/1.1\" 401 98",
        "malformed log line"
    ]

    for i, line in enumerate(test_lines):
        print(f"Test Line {i + 1}: {line}")
        parsed = parse_log_line(line)
        print("Parsed Output:", parsed, "\n")

def filter_and_classify_logs(parsed_logs):
    """
    Classify parsed logs into categories and count them.
    """
    errors = []
    successes = []
    suspicious = []
    malformed = 0

    for entry in parsed_logs:
        if not entry:
            malformed += 1
            continue

        status = entry["status"]
        if 400 <= status <= 599:
            errors.append(entry)
        elif 200 <= status <= 299:
            successes.append(entry)

        if "/admin" in entry["resource"] or "/wp-admin" in entry["resource"]:
            suspicious.append(entry)

    return {
        "errors": errors,
        "successes": successes,
        "suspicious": suspicious,
        "malformed": malformed
    }


# Print classification counts
def print_classification_counts(classifications):
    """
    Print the number of lines in each category.
    """
    print("\n--- Log Classification Counts ---")
    print(f"Total logs parsed: {sum(len(v) for k, v in classifications.items() if k != 'malformed') + classifications['malformed']}")
    print(f"Malformed logs: {classifications['malformed']}")
    print(f"Error logs: {len(classifications['errors'])}")
    print(f"Successful logs: {len(classifications['successes'])}")
    print(f"Suspicious logs: {len(classifications['suspicious'])}")

def analyze_logs(parsed_logs):
    """
    Analyze parsed logs to extract meaningful statistics.
    """
    unique_ips = set()
    requests_per_ip = Counter()
    resource_requests = Counter()
    method_counts = Counter()
    hourly_traffic = Counter()

    for entry in parsed_logs:
        if not entry:
            continue

        unique_ips.add(entry["ip"])
        requests_per_ip[entry["ip"]] += 1
        resource_requests[entry["resource"]] += 1
        method_counts[entry["method"]] += 1

        timestamp = datetime.strptime(entry["timestamp"], "%d/%b/%Y:%H:%M:%S %z")
        hourly_traffic[timestamp.strftime("%d/%b/%Y:%H")] += 1

    busiest_hour = hourly_traffic.most_common(1)[0] if hourly_traffic else None

    return {
        "total_requests": len(parsed_logs),
        "unique_ips": len(unique_ips),
        "top_ips": requests_per_ip.most_common(3),
        "most_requested_resource": resource_requests.most_common(1),
        "method_distribution": method_counts,
        "busiest_hour": busiest_hour
    }

def generate_report(log_analysis, classifications):
    """
    Generate a summary report of the log analysis and classifications.
    """
    report = ["--- SUMMARY REPORT ---",
              f"Total parsed log entries: {log_analysis['total_requests']}",
              f"Malformed entries: {classifications['malformed']}",
              f"Distinct IP addresses: {log_analysis['unique_ips']}",
              "Top 3 IP addresses by request count:"]

    for ip, count in log_analysis['top_ips']:
        report.append(f"  {ip} : {count} requests")

    most_requested = log_analysis['most_requested_resource']
    if most_requested:
        report.append(f"Most requested resource: {most_requested[0][0]} ({most_requested[0][1]} requests)")

    report.append("HTTP method usage:")
    for method, count in log_analysis['method_distribution'].items():
        report.append(f"  {method} = {count}")

    if log_analysis['busiest_hour']:
        report.append(f"Busiest hour: {log_analysis['busiest_hour'][0]} with {log_analysis['busiest_hour'][1]} requests")

    return "\n".join(report)

def advanced_analysis(parsed_logs):
    """
    Perform advanced analysis like grouping IPs by Class C network and detecting repeated error attempts.
    """
    class_c_networks = defaultdict(list)
    repeated_error_ips = Counter()

    for entry in parsed_logs:
        if not entry:
            continue

        # Group by Class C network
        ip_parts = entry["ip"].split(".")
        if len(ip_parts) >= 3:
            class_c_network = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.*"
            class_c_networks[class_c_network].append(entry["ip"])

        # Detect repeated error attempts
        if 400 <= entry["status"] <= 599:
            repeated_error_ips[entry["ip"]] += 1

    return {
        "class_c_networks": {k: len(set(v)) for k, v in class_c_networks.items()},
        "repeated_error_ips": repeated_error_ips.most_common()
    }

def save_to_csv(parsed_logs, output_file):
    """
    Save the parsed logs to a CSV file.
    """
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["ip", "ident", "user", "timestamp", "method", "resource", "protocol", "status", "size"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in parsed_logs:
            if entry:
                writer.writerow(entry)

if __name__ == "__main__":
    log_file_path = "../logs/access-1.log"
    csv_output_path = "../logs/parsed_logs.csv"

    try:
        with open(log_file_path, "r") as log_file:
            raw_logs = log_file.readlines()

        parsed_logs = [parse_log_line(line) for line in raw_logs]
        classifications = filter_and_classify_logs(parsed_logs)
        log_analysis = analyze_logs(parsed_logs)
        advanced_analysis_results = advanced_analysis(parsed_logs)

        # Verification step
        print("\n--- Verification of Parsing Function ---\n")
        verify_parsing_function()

        # Print classification counts
        print_classification_counts(classifications)

        report = generate_report(log_analysis, classifications)
        print(report)

        # Print advanced analysis results
        print("\n--- Advanced Analysis ---")
        print("Class C Networks:")
        for network, count in advanced_analysis_results["class_c_networks"].items():
            print(f"  {network}: {count} unique IPs")
        print("Repeated Error Attempts:")
        for ip, count in advanced_analysis_results["repeated_error_ips"]:
            print(f"  {ip}: {count} errors")

        # Save parsed logs to CSV
        save_to_csv(parsed_logs, csv_output_path)
        print(f"\nParsed logs saved to {csv_output_path}")



    except FileNotFoundError:
        print(f"Error: File not found: {log_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
