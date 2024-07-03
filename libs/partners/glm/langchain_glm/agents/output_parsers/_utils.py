# Function to find positions of object() instances
def find_object_positions(log_chunk, obj):
    return [i for i, x in enumerate(log_chunk) if x == obj]


# Function to concatenate segments based on object positions
def concatenate_segments(log_chunk, positions):
    segments = []
    start = 0
    for pos in positions:
        segments.append("".join(map(str, log_chunk[start:pos])))
        start = pos + 1
    return segments
