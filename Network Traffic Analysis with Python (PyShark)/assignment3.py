import pyshark

# Optional: Set TShark path if needed
# import pyshark.tshark.tshark
# pyshark.tshark.tshark.get_tshark_path = lambda: "C:\\Program Files\\Wireshark\\tshark.exe"

# Load HTTP packets only
cap = pyshark.FileCapture("http.cap", display_filter="http.request")

print("HTTP Requests Found:")

for packet in cap:
    try:
        host = packet.http.host
        uri = packet.http.request_uri
        full_url = f"http://{host}{uri}"
        
        print(f"Host     : {host}")
        print(f"URI      : {uri}")
        print(f"Full URL : {full_url}")
        print("-" * 40)
        
    except AttributeError:
        continue
