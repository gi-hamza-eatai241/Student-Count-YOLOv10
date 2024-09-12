import multiprocessing
import time

import cv2
from ultralytics import YOLO


def extend_line(line_start, line_end, img_width, img_height):
    """
    Extend a line to the full width or height of the image while maintaining its original angle.

    Args:
        line_start (tuple): Starting point of the line (x, y).
        line_end (tuple): Ending point of the line (x, y).
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        tuple: Extended start and end points of the line.
    """
    # Calculate slope and intercept of the line
    delta_x = line_end[0] - line_start[0]
    delta_y = line_end[1] - line_start[1]

    if delta_x == 0:  # Vertical line
        extended_start = (line_start[0], 0)
        extended_end = (line_end[0], img_height)
    elif delta_y == 0:  # Horizontal line
        extended_start = (0, line_start[1])
        extended_end = (img_width, line_end[1])
    else:
        # Calculate the slope
        slope = delta_y / delta_x
        intercept = line_start[1] - slope * line_start[0]

        # Extend to the left and right edges of the frame
        x_start = 0
        y_start = int(intercept)

        x_end = img_width
        y_end = int(slope * img_width + intercept)

        # Extend to the top and bottom edges if necessary
        if y_start < 0:
            y_start = 0
            x_start = int(-intercept / slope)

        if y_end > img_height:
            y_end = img_height
            x_end = int((img_height - intercept) / slope)

        extended_start = (x_start, y_start)
        extended_end = (x_end, y_end)

    return extended_start, extended_end

def process_camera(camera_id, video_source, output_video_path, line_coordinates):
    """
    Process a single camera feed for object detection, tracking, and counting.

    Args:
        camera_id (int): Unique identifier for the camera.
        video_source (str or int): Path to the video file or camera index.
        output_video_path (str): Path to save the output video.
    """
    # Load YOLO model
    model = YOLO("yolov10x.engine", task="detect")

    # Open video source
    cap = cv2.VideoCapture(video_source)
    assert cap.isOpened(), f"Error opening video source {video_source}"

    # Get video parameters
    orig_width, orig_height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define reduced processing resolution and output resolution
    process_width, process_height = 640, 360
    output_width, output_height = 1920, 1080

    # Define the line points from the selected points in the original resolution
    line_start, line_end = line_coordinates # (2, 448) # (745, 719)
    # line_end = (1279, 439) # (542, 3)

    # Extend the line based on its orientation
    extended_line_start, extended_line_end = extend_line(line_start, line_end, process_width,
                                                         process_height)

    # Video writer with the output resolution
    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height)
    )

    # Initialize counters
    count_in = 0
    count_out = 0
    actual_count_out = 0
    people_inside = 0

    # Dictionaries and sets for tracking
    last_positions = {}

    def is_crossing_line(p1, p2, line_point_1, line_point_2):
        """
        Check if object is crossing the line by comparing the sign of the determinant of the two vectors.
        """
        d1 = (line_point_2[1] - line_point_1[1]) * p1[0] - (line_point_2[0] - line_point_1[0]) * p1[1] + line_point_2[0] * line_point_1[1] - line_point_2[1] * line_point_1[0]
        d2 = (line_point_2[1] - line_point_1[1]) * p2[0] - (line_point_2[0] - line_point_1[0]) * p2[1] + line_point_2[0] * line_point_1[1] - line_point_2[1] * line_point_1[0]
        # print(d1, d2)
        return d1 * d2 < 0  # Check if signs are opposite, which means it crossed the line

    frame_count = 0
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print(f"Camera {camera_id}: Video processing completed or no frame.")
            break

        frame_count += 1

        if frame_count % 2 == 0:
            continue

        # Resize frame for processing
        im0_resized = cv2.resize(im0, (process_width, process_height))

        # Run object detection and tracking
        results = model.track(im0_resized, persist=True, show=False, classes=[0], verbose=False)

        # Check if any detections were made
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Check if box.id exists and is not None
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id.item())

                    # Get the center coordinates of the bounding box in the reduced resolution
                    x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    confidence = box.conf.item()  # Confidence score

                    # Draw the bounding box and center point on the resized frame
                    cv2.rectangle(im0_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(im0_resized, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(im0_resized, f"({center_x}, {center_y})", (center_x + 10 , center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Display the track_id and confidence near the bounding box
                    cv2.putText(
                        im0_resized,
                        f"ID: {track_id}, Conf: {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    # Tracking and counting logic
                    if track_id in last_positions:
                        # print(track_id)
                        last_position = last_positions[track_id]

                        # Check if the object crossed the line
                        if is_crossing_line(last_position, (center_x, center_y), line_start, line_end):
                            # Object moving right to left (out)
                            # if last_position[1] < center_y:
                            if last_position[0] < center_x:
                                print(f"Camera {camera_id}: Object {track_id} went out")
                                actual_count_out += 1
                                if people_inside > 0:
                                    count_out += 1
                            # Object moving left to right (in)
                            else:
                                print(f"Camera {camera_id}: Object {track_id} came in")
                                count_in += 1

                    # Update last position
                    last_positions[track_id] = (center_x, center_y)

        people_inside = max(0, count_in - count_out)

        # Display counts on the resized frame
        cv2.putText(im0_resized, f"IN: {count_in}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        cv2.putText(im0_resized, f"OUT: {actual_count_out}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
        cv2.putText(im0_resized, f"INSIDE: {people_inside}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 1)

        # Draw the counting line on the resized frame
        # cv2.line(im0_resized, line_start, line_end, (0, 0, 255), 2)

        cv2.line(im0_resized, extended_line_start, extended_line_end, (0, 0, 255), 1)

        # Show the result in real-time
        cv2.imshow(f"Camera {camera_id} - Real-Time Counting", im0_resized)

        # Resize back to output resolution for saving
        im0_output = cv2.resize(im0_resized, (output_width, output_height))

        # Write frame to video
        video_writer.write(im0_output)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Camera {camera_id}: Exiting on user command.")
            break

    # Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Camera {camera_id}: Processing finished.")


if __name__ == "__main__":
    # List of camera sources (video files or camera indices)
    camera_sources = [
        # ("Video 1", "SuperNova-CCTV-Side-Angled.mp4", "SuperNova-CCTV-Side-Angled_Result.mp4", [(742, 1), (822, 717)]),
        # ("tapo-cam-1", "rtsp://grilsquad:grilsquad@192.168.3.41:554/stream1", "output.mp4", [(728, 1), (592, 719)]),
        ("tapo-cam-1", "rtsp://grilsquad:grilsquad@192.168.3.41:554/stream1", "output_AI-Accelerator.mp4", [(253, 168), (296, 358)])#[(282, 7), (349, 337)])# [(357, 4), (398, 359)]) #[(237, 1), (438, 358)]),
        # Add more cameras as needed
    ]

    # Create a list to hold processes
    processes = []

    # Start a process for each camera
    for idx, (camera_name, input_path, output_path, line_points) in enumerate(camera_sources):
        p = multiprocessing.Process(
            target=process_camera,
            args=(camera_name, input_path, output_path, line_points),
        )
        processes.append(p)
        p.start()
        time.sleep(1)  # Slight delay to avoid overlapping prints

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All camera processes have completed.")
