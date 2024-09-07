import cv2
from ultralytics import YOLO
import multiprocessing
import time

lock = multiprocessing.Lock()

def process_camera(camera_id, video_source, line_y, output_video_path, people_inside, is_entry_gate):
    """
    Process a single camera feed for object detection, tracking, and counting.

    Args:
        camera_id (int): Unique identifier for the camera.
        video_source (str or int): Path to the video file or camera index.
        line_y (int): Y-coordinate of the counting line.
        output_video_path (str): Path to save the output video.
        people_inside (multiprocessing.Value): Shared counter for people inside the room.
        is_entry_gate (bool): True if this camera is for the entry gate, False if for the exit gate.
    """
    # Load YOLO model
    model = YOLO("yolov8x.engine", task="detect")

    # Open video source
    cap = cv2.VideoCapture(video_source)
    assert cap.isOpened(), f"Error opening video source {video_source}"

    # Get video parameters
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define line points for counting
    frame_width = w
    line_points = [(0, line_y), (frame_width - 1, line_y)]

    # Video writer
    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # Dictionaries and sets for tracking
    last_positions = {}

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print(f"Camera {camera_id}: Video processing completed or no frame.")
            break

        # Run object detection and tracking
        results = model.track(im0, persist=True, show=False, classes=[0])

        # Check if any detections were made
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Check if box.id exists and is not None
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id.item())

                    # Get the center y-coordinate of the bounding box
                    x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    confidence = box.conf.item()  # Confidence score

                    # Draw the bounding box and center point
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(im0, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Display the track_id and confidence near the bounding box
                    cv2.putText(
                        im0,
                        f"ID: {track_id}, Conf: {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    # Tracking and counting logic
                    if track_id in last_positions:
                        last_y = last_positions[track_id]

                        # Entry gate logic (object moving downwards)
                        if is_entry_gate and last_y < line_y <= center_y:
                            print(f"Camera {camera_id}: Object {track_id} came in")
                            with lock:  # Lock automatically handled by Value
                                people_inside.value += 1

                        # Exit gate logic (object moving upwards)
                        elif not is_entry_gate and last_y > line_y >= center_y:
                            print(f"Camera {camera_id}: Object {track_id} went out")
                            with lock:  # Lock automatically handled by Value
                                if people_inside.value > 0:
                                    people_inside.value -= 1

                    # Update last position
                    last_positions[track_id] = center_y

        # Display counts
        cv2.putText(im0, f"INSIDE: {people_inside.value}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Draw the counting line
        cv2.line(im0, line_points[0], line_points[1], (0, 0, 255), 2)

        # Write frame to video
        video_writer.write(im0)

        # # Show the result in real-time
        # im0 = cv2.resize(im0, (640, 360))
        # cv2.imshow(f"Camera {camera_id} - Real-Time Counting", im0)

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
    # Shared variable to track people inside the room
    with multiprocessing.Manager() as manager:
        people_inside = manager.Value('i', 0)  # 'i' means integer

        # List of camera sources (video files or camera indices)
        camera_sources = [
            ("Camera 1", "entry_video.mp4", 400, "output_entry.avi", True),  # Entry gate
            ("Camera 2", "exit_video.mp4", 400, "output_exit.avi", False),   # Exit gate
               # Add more cameras as needed
        ]

        # Create a list to hold processes
        processes = []

        # Start a process for each camera
        for idx, (camera_name, video_source, line_y, output_video_path, is_entry_gate) in enumerate(camera_sources):
            p = multiprocessing.Process(
                target=process_camera,
                args=(camera_name, video_source, line_y, output_video_path, people_inside, is_entry_gate)
            )
            processes.append(p)
            p.start()
            time.sleep(1)  # Slight delay to avoid overlapping prints

        # Wait for all processes to finish
        for p in processes:
            p.join()

        print(f"Final count of people inside: {people_inside.value}")
        print("All camera processes have completed.")
