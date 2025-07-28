import datetime
import time

# Get the start time
start_time = datetime.datetime.now()

# Calculate end time (10 minutes from start)
end_time = start_time + datetime.timedelta(minutes=1)

# Open file in append mode
with open('time_log.txt', 'a') as f:
    # Write the start time
    f.write(f"Start time: {start_time}\n")

    # Keep running until we reach 10 minutes
    while datetime.datetime.now() < end_time:
        # Get current time and write to file
        current_time = datetime.datetime.now()
        f.write(f"{current_time}\n")

        # Wait 30 seconds before next iteration
        time.sleep(30)

    # Write the end time
    f.write(f"End time: {datetime.datetime.now()}\n")
    f.write("-" * 50 + "\n")  # Separator line