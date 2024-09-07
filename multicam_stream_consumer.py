import threading
from student_count import batched_frame_student_count


lock = threading.Lock()
BATCH_SIZE = 16
LIVE_STREAM_BUFFER_SIZE = 2048
LIVE_STREAM_BUFFER_PURGE_SIZE = 64
NUMBER_OF_THREADS = 2

def student_count(shared_buffer):
    """
    This function gets frames from the shared buffer and collects them in batches for face recognition.

    Parameters:
        shared_buffer: A shared buffer that contains frames from all the cameras
    """
    while True:
        # Apply thread lock to the shared buffer
        with lock:
            # frame_accumulation_start_time = time.time()
            # if the shared buffer to the corresponding camera is empty, then continue
            if shared_buffer.qsize() < BATCH_SIZE:
                continue

            batched_frame_buffer = []
            while len(batched_frame_buffer) != BATCH_SIZE:
                camera_elements = shared_buffer.get()
                batched_frame_buffer.append(camera_elements)
            # frame_accumulation_end_time = time.time()
            # frame_accumulation_total_time = frame_accumulation_end_time - frame_accumulation_start_time
            # print(
            #     f"Time taken to accumulate frames for batch of size {BATCH_SIZE} is "
            #     f"{frame_accumulation_total_time:>20.9f} seconds"
            # )
            # multicam_server_logger.info(
            #     f"Time taken to accumulate frames for batch of size {BATCH_SIZE} is "
            #     f"{frame_accumulation_total_time:>20.9f} seconds"
            # )

        # extract batch of frames and camera names from the batched_frame_buffer
        # batch_extraction_start_time = time.time()
        batch_of_frames = [batch_elements[0] for batch_elements in batched_frame_buffer]
        batch_of_cam_names = [batch_elements[1] for batch_elements in batched_frame_buffer]
        batch_of_cam_ips = [batch_elements[2] for batch_elements in batched_frame_buffer]
        # batch_extraction_end_time = time.time()
        # batch_extraction_total_time = batch_extraction_end_time - batch_extraction_start_time
        # multicam_server_logger.info(
        #     f"Time taken to extract batch of size {BATCH_SIZE} is {batch_extraction_total_time:>20.9f} seconds"
        # )

        # recognition_start_time = time.time()
        # Perform batched face recognition on the frames in the buffer
        batched_frame_student_count(
            batch_of_frames=batch_of_frames,
            batch_of_cam_names=batch_of_cam_names,
            batch_of_cam_ips=batch_of_cam_ips,
        )
        # recognition_end_time = time.time()
        # recognition_total_time = recognition_end_time - recognition_start_time
        # multicam_server_logger.info(
        #     f"Time taken to process batch of size {BATCH_SIZE} is {recognition_total_time:>20.9f} seconds"
        # )

        # Delete batch_of_frames
        del batch_of_frames

        frames_buffer_size = shared_buffer.qsize()
        # print(f"THREAD: {thread_id} CURRENT BUFFER SIZE: {frames_buffer_size}")
        if frames_buffer_size > LIVE_STREAM_BUFFER_SIZE:
            print(f"Frames buffer size {frames_buffer_size}. Purging frames_buffer...")
            for _ in range(LIVE_STREAM_BUFFER_PURGE_SIZE):
                shared_buffer.get()

        # consumption_end_time = time.time()
        # total_consumption_time = consumption_end_time - consumption_start_time
        # multicam_server_logger.info(
        #     f"Total time taken by consumer to consume the whole batch is {total_consumption_time:>20.9f} seconds"
        # )

def consumer_main(shared_buffer):
    """
    This function starts the face recognition process.

    Parameters:
        shared_buffer: A shared buffer that contains frames from all the cameras
    """
    try:
        student_count(shared_buffer)

    except KeyboardInterrupt:
        print("EXITING THE PROGRAM...")