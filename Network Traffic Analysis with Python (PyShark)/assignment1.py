import pyshark

# Optional: Set TShark path if needed
# import pyshark.tshark.tshark
# pyshark.tshark.tshark.get_tshark_path = lambda: "C:\\Program Files\\Wireshark\\tshark.exe"

# Load .pcap file
cap = pyshark.FileCapture("http.cap")

print("First 5 Packets:")
for i, packet in enumerate(cap):
    if i == 5:
        break
    try:
        print(f"Packet {i+1}")
        print(f"  • Number          : {packet.number}")
        print(f"  • Length          : {packet.length}")
        print(f"  • Highest Layer   : {packet.highest_layer}")
        print(f"  • Transport Layer : {packet.transport_layer}")
        print("-" * 50)
    except AttributeError as e:
        print(f"Skipping packet {i + 1}: {e}")
