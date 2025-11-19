"""
Latency compensation and timestamp alignment.
"""
def latency_compensate(tick_timestamp, avg_network_delay):
    return tick_timestamp - avg_network_delay
