import unittest
from collections import Counter

from parser.web_log_parser import parse_log_line, filter_and_classify_logs, analyze_logs, \
    advanced_analysis


class TestWebLogParser(unittest.TestCase):

    def test_parse_log_line(self):
        test_line = "123.45.67.89 - - [12/Dec/2024:06:25:17 +0000] \"GET /index.html HTTP/1.1\" 200 5123"
        expected_result = {
            "ip": "123.45.67.89",
            "ident": "-",
            "user": "-",
            "timestamp": "12/Dec/2024:06:25:17 +0000",
            "method": "GET",
            "resource": "/index.html",
            "protocol": "HTTP/1.1",
            "status": 200,
            "size": "5123"
        }
        self.assertEqual(parse_log_line(test_line), expected_result)

        malformed_line = "malformed log line"
        self.assertIsNone(parse_log_line(malformed_line))

    def test_filter_and_classify_logs(self):
        parsed_logs = [
            {
                "ip": "123.45.67.89",
                "ident": "-",
                "user": "-",
                "timestamp": "12/Dec/2024:06:25:17 +0000",
                "method": "GET",
                "resource": "/index.html",
                "protocol": "HTTP/1.1",
                "status": 200,
                "size": "5123"
            },
            {
                "ip": "55.66.77.88",
                "ident": "-",
                "user": "admin",
                "timestamp": "12/Dec/2024:06:25:18 +0000",
                "method": "POST",
                "resource": "/admin",
                "protocol": "HTTP/1.1",
                "status": 401,
                "size": "98"
            },
            None
        ]

        classifications = filter_and_classify_logs(parsed_logs)
        self.assertEqual(len(classifications["errors"]), 1)
        self.assertEqual(len(classifications["successes"]), 1)
        self.assertEqual(len(classifications["suspicious"]), 1)
        self.assertEqual(classifications["malformed"], 1)

    def test_analyze_logs(self):
        parsed_logs = [
            {
                "ip": "123.45.67.89",
                "ident": "-",
                "user": "-",
                "timestamp": "12/Dec/2024:06:25:17 +0000",
                "method": "GET",
                "resource": "/index.html",
                "protocol": "HTTP/1.1",
                "status": 200,
                "size": "5123"
            },
            {
                "ip": "123.45.67.89",
                "ident": "-",
                "user": "-",
                "timestamp": "12/Dec/2024:07:25:17 +0000",
                "method": "GET",
                "resource": "/home.html",
                "protocol": "HTTP/1.1",
                "status": 200,
                "size": "1024"
            }
        ]

        analysis = analyze_logs(parsed_logs)
        self.assertEqual(analysis["total_requests"], 2)
        self.assertEqual(analysis["unique_ips"], 1)
        self.assertEqual(analysis["top_ips"], [("123.45.67.89", 2)])
        self.assertEqual(analysis["most_requested_resource"], [("/index.html", 1)])
        self.assertEqual(analysis["method_distribution"], Counter({"GET": 2}))

    def test_advanced_analysis(self):
        parsed_logs = [
            {
                "ip": "123.45.67.89",
                "ident": "-",
                "user": "-",
                "timestamp": "12/Dec/2024:06:25:17 +0000",
                "method": "GET",
                "resource": "/index.html",
                "protocol": "HTTP/1.1",
                "status": 404,
                "size": "5123"
            },
            {
                "ip": "123.45.67.90",
                "ident": "-",
                "user": "-",
                "timestamp": "12/Dec/2024:07:25:17 +0000",
                "method": "POST",
                "resource": "/login",
                "protocol": "HTTP/1.1",
                "status": 500,
                "size": "2048"
            }
        ]

        advanced_results = advanced_analysis(parsed_logs)
        self.assertEqual(advanced_results["class_c_networks"], {"123.45.67.*": 2})
        self.assertEqual(advanced_results["repeated_error_ips"], [("123.45.67.89", 1), ("123.45.67.90", 1)])

if __name__ == "__main__":
    unittest.main()
