from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


def format_pysrt(pysrt_object: Iterable) -> List[Dict[str, Any]]:
    """
    Format pysrt object to a specific format.

    Parameters
    ----------
    pysrt_object : Subripefile

    Returns
    -------

    transcritpts :
    [
        {
            'text':str
            'start': float
            'duration':float
        }
    ]

    """
    total_seconds = lambda t: (
        t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1000000
    )
    transcripts = []
    for p in pysrt_object:
        text = p.text
        start = total_seconds(p.start.to_time())
        end = total_seconds(p.end.to_time())
        duration = end - start
        transcripts += [
            {
                "text": text,
                "start": start,
                "duration": duration,
            }
        ]
    return transcripts


def chunk_transcripts(
    transcripts: List[Dict[str, Any]], duration: Optional[float]
) -> List[Dict[str, Any]]:
    """
    Chunk transcripts with a given duration.

    Parameters
    ----------

    transcritpts :
    [
        {
            'text':str
            'start': float
            'duration':float
        }
    ]

    duration : float

    Returns
    -------

    chunks :
    [
        {
            'text':str
            'start': float
            'duration':float
        }
    ]

    """
    if duration == None:
        return []
    chunks = []
    p1 = 0
    n = len(transcripts)
    """
    Two pointer window approach extend the second pointer
    as much as possible keeping the error ( difference from
    duration) minimum by choosing or not choosing to add 
    """
    while p1 < n:
        curr_chunk = transcripts[p1].copy()
        curr_dur = transcripts[p1]["duration"]
        curr_err = abs(curr_dur - duration)
        p2 = p1 + 1
        while p2 < n:
            curr_dur += deepcopy(transcripts[p2]["duration"])
            new_err = abs(curr_dur - duration)
            if new_err <= curr_err:
                curr_chunk.update(
                    {
                        "text": curr_chunk["text"] + " " + transcripts[p2]["text"],
                        "duration": round(curr_dur, 2),
                    }
                )
                curr_err = new_err
                p2 += 1
            else:
                break
        chunks += [curr_chunk]
        p1 = p2
    return chunks
