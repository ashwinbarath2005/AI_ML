import pyshark
from collections import Counter

# Optional: Set TShark path if needed
# import pyshark.tshark.tshark
# pyshark.tshark.tshark.get_tshark_path = lambda: "C:\\Program Files\\Wireshark\\tshark.exe"

# Load only DNS packets
cap = pyshark.FileCapture("http.cap", display_filter="dns")

dns_counter = Counter()

print("Tracking DNS Queries...")

for packet in cap:
    try:
        domain = packet.dns.qry_name
        dns_counter[domain] += 1
    except AttributeError:
        continue

print("\nTop 10 Most Queried Domains:")
for domain, count in dns_counter.most_common(10):
    print(f"{domain} -> {count} times")
