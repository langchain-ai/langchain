import os
import datetime

def mission_control_log(event):
    """
    Logs events to both console and a file for 'Visual Mission Control'.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the event for logging
    if isinstance(event, dict):
        log_message = f"[{timestamp}] EVENT: {event}\n"
        # Check for specific keys to make logs more readable
        if "messages" in event:
            # If it's a message update, log the last message
            last_msg = event["messages"][-1]
            log_message += f"   > MESSAGE: {last_msg}\n"
    else:
        log_message = f"[{timestamp}] RAW: {event}\n"

    # Write to file
    with open("swarm.log", "a", encoding="utf-8") as f:
        f.write(log_message)
    
    # Also print to console (optional, but good for immediate feedback)
    print(log_message.strip())
