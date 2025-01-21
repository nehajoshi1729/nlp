# Web Server Log Parser

## How to Run the Code

1. **Prerequisites:**
   - Ensure Python 3.x is installed on your system.
   - This script does not rely on external libraries beyond Python's standard library.

2. **Steps to Run:**
   - The log file (`access-1.log`) is placed in the logs/ directory.
   - The web parser script is placed in the /parser directory.
   - Run the script using the command :
     ```
     cd parser
     python web_log_parser.py
     ```
   - The script will:
     - Parse the log file.
     - Filter and classify the entries.
     - Perform analysis and generate a summary report.
     - Save the parsed logs to a CSV file in logs/ (`parsed_logs.csv`).

3. **Output:**
   - Console output includes:
     - Summary report of log analysis.
     - Classification counts (e.g., total logs, errors, successes).
     - Advanced analysis results (Class C networks and repeated error attempts).
   - CSV file containing structured parsed logs.

---

## Format of the Regular Expression

The regular expression used for parsing log lines is:
```regex
^(\S+) (\S+) (\S+) \[(.*?)\] \"(\S+) (.*?) (\S+)\" (\d{3}) (\S+)$
```
### Explanation of the Groups:
1. **IP Address or Hostname:** Captures the client IP address or hostname.
2. **Ident:** Captures the `ident` field (commonly `-`).
3. **UserID:** Captures the user ID (commonly `-` or a username).
4. **Timestamp:** Captures the timestamp enclosed in square brackets (e.g., `12/Dec/2024:06:25:17 +0000`).
5. **HTTP Method:** Captures the request method (e.g., `GET`, `POST`).
6. **Requested Resource:** Captures the requested path (e.g., `/index.html`).
7. **Protocol:** Captures the HTTP protocol (e.g., `HTTP/1.1`).
8. **Status Code:** Captures the response status code (e.g., `200`, `404`).
9. **Size:** Captures the size of the response in bytes or `-` if unavailable.

---

## Assumptions

1. **Log Format:**
   - Assumes logs follow a Common Log Format (CLF) or a slightly modified version.
   - Malformed lines that do not match the regex are ignored.

2. **Error Code Ranges:**
   - **Client Errors:** Status codes between `400-499`.
   - **Server Errors:** Status codes between `500-599`.

3. **Suspicious Paths:**
   - Any request targeting paths like `/admin` or `/wp-admin` is flagged as suspicious.

4. **Advanced Analysis:**
   - **Class C Networks:** IPs are grouped by the first three octets (e.g., `123.45.67.*`).
   - **Repeated Error Attempts:** IPs with multiple error responses are tracked and reported.

5. **CSV Output:**
   - The CSV file includes fields for IP, ident, user, timestamp, method, resource, protocol, status, and size.

---


