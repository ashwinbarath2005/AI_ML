import pyshark

interface_name = 'Wi-Fi'  # Replace if needed

print("Capturing packets on:", interface_name)

capture = pyshark.LiveCapture(interface=interface_name, display_filter='tcp.port == 80')

packet_count = 0
for packet in capture.sniff_continuously():
    try:
        print(f"\nPacket {packet_count + 1}")
        print(f"  • Source: {packet.ip.src}")
        print(f"  • Destination: {packet.ip.dst}")
        print(f"  • Transport: {packet.transport_layer}")
        print(f"  • Length: {packet.length}")
        print("-" * 40)
        packet_count += 1
        if packet_count == 10:
            break
    except AttributeError:
        continue
