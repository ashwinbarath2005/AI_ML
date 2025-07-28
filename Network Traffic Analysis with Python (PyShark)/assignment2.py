import pyshark
from collections import Counter

# Optional: Set TShark path if needed
# import pyshark.tshark.tshark
# pyshark.tshark.tshark.get_tshark_path = lambda: "C:\\Program Files\\Wireshark\\tshark.exe"

cap = pyshark.FileCapture("http.cap")

ip_counter = Counter()

print("Analyzing packets to count top IP Talkers...")

for packet in cap:
    try:
        if 'IP' in packet:
            src_ip = packet.ip.src
            ip_counter[src_ip] += 1
    except AttributeError:
        continue  # skip if IP layer missing

print("\nTop 5 IP Talkers:")
for ip, count in ip_counter.most_common(5):
    print(f"{ip} -> {count} packets")
