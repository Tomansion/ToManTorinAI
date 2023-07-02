import tensorflow as tf

# Directory containing the event files
event_files_dir = 'logs/self_fighter_PPO_3_30M_0/'

# Output file path for the merged event file
merged_file_path = 'logs/merged_events.out.tfevents'

# Get a list of all event files in the directory
event_files = tf.io.gfile.glob(event_files_dir + 'events.out.tfevents.*')

# Create an EventFileWriter to write the merged events
with tf.compat.v1.summary.FileWriter(merged_file_path) as writer:
    # Iterate over each event file
    for event_file in event_files:
        # Create an EventFileReader for the current event file
        reader = tf.compat.v1.summary.FileWriter(event_file)

        # Iterate over each event in the event file
        for event in reader:
            # Write the event to the merged event file
            writer.add_event(event)

print("Merged event file created successfully!")
