import threading
import time
import cv2

# A list of all the active cameras
cameras = []

FRAME_RATE_FACTOR = 3
IP_CAM_REINIT_WAIT_DURATION = 10
CAM_USERNAME = "grilsquad"
CAM_PASSWORD = "grilsquad"
IP_CAMS = {
    "cam-1": "video.mp4",
    "cam-2": "video.mp4"
}


class IPCamera:
    """A class to represent an IP camera"""

    def __init__(self, cam_name: str, cam_ip: str, shared_buffer):
        """Initialize the camera"""
        self.frame = None
        self.shared_buffer = shared_buffer
        self.cam_name = cam_name
        self.cam_ip = cam_ip
        # # timestamp before capturing video stream
        # tick = time.time()
        # initialize the video camera stream and read the first frame
        self.stream = cv2.VideoCapture(cam_ip if cam_ip.startswith("http") else f"rtsp://{CAM_USERNAME}:{CAM_PASSWORD}@{cam_ip}:554/stream1")
        # we need to read the first frame to initialize the stream
        # _, _ = self.stream.read()
        self.grabbed, _ = self.stream.read()
        # store whether the camera stream was initialized successfully
        self.is_initialized = self.grabbed
        # set the flag to process the frame
        self.process_this_frame = True
        # initialize a frame counter
        self.frame_counter = 0
        # buffers for darkness detection
        self.light_intensity_detection_buffer = []
        self.darkness_buffer = []
        # initial darkness status
        self.too_dark = False
        # Flag to track if adequate lighting message is sent
        self.adequate_lighting_message_sent = True
        # Flag to track if inadequate lighting message is sent
        self.inadequate_lighting_message_logged = False

        if not self.grabbed:
            print(
                f"Camera stream from {self.cam_name} (url: {self.cam_ip})) unable to initialize"
            )
        else:
            # # timestamp once the frame is grabbed successfully
            # tock = time.time()
            # # total time to grab and initialize the stream
            # time_taken = tock - tick
            # print(f"Time taken to grab and initialize the camera stream is {time_taken:>20.9f} seconds")
            # multicam_server_logger.info(
            #     f"Time taken to grab and initialize the camera stream is {time_taken:>20.9f} seconds"
            # )
            print(
                f"Camera stream from {self.cam_name} (url: {self.cam_ip}) initialized"
            )

    def _read_one_frame(self):
        """Reads a frame from the camera"""
        self.grabbed, self.frame = self.stream.read()

    def _read_and_discard_frame(self):
        """Reads and discards one frame"""
        _, _ = self.stream.read()

    def release(self):
        """Releases the camera stream"""
        self.stream.release()

    def place_frame_in_buffer(self):
        """Places the frame in the buffer"""
        # frame_processed = None
        if self.process_this_frame:
            self._read_one_frame()
            if not self.grabbed:
                # if the frame was not grabbed, then we have reached the end of the stream
                print(
                    f'Could not read a frame from the camera stream from {self.cam_name} (url: {self.cam_ip})). '
                    f'Releasing the stream...')
                self.release()
                self.is_initialized = False
            else:
                # resize the frame if the frame size is larger than the frame size specified in parameters.py
                self.frame = cv2.resize(self.frame, (1920, 1080))

                self.shared_buffer.put((self.frame, self.cam_name, self.cam_ip))

            # # set the flag to True since the frame was processed
            # frame_processed = True

        else:
            self._read_and_discard_frame()

            # # set the flag to False since the frame was not processed
            # frame_processed = False

        # toggle the flag to process alternate frames to improve the performance
        self.frame_counter += 1
        if self.frame_counter % FRAME_RATE_FACTOR == 0:
            self.process_this_frame = True
        else:
            self.process_this_frame = False

        # # return the flag
        # return frame_processed


def create_camera(cam_name: str, cam_ip: str, shared_buffer):
    """
    Creates a camera object and places the frames in the buffer

    Args:
        cam_name (str): name of the camera
        cam_ip (str): url of the camera
        shared_buffer: shared memory space to store the pre-processed frames

    Returns:
        None
    """
    global cameras

    cam = IPCamera(cam_name, cam_ip, shared_buffer)
    cameras.append(cam)
    # Place the frames in the buffer until the end of the camera stream is reached

    # # Initialize a start time and a frame counter
    # fps_start_time = time.time()
    # fps_frame_counter = 0
    # batch_start_time = time.time()
    # batch_frame_counter = 0

    while True:
        if cam.is_initialized:
            # # Message to be sent
            # message = "Camera connection successful"
            # # Send Kafka message
            # cam.send_kafka_message(message)
            try:
                cam.place_frame_in_buffer()
                # if cam.place_frame_in_buffer():
                #     # increment the frame counter
                #     fps_frame_counter += 1
                #     batch_frame_counter += 1
                #     if batch_frame_counter == BATCH_SIZE:
                #         batch_end_time = time.time()
                #         batch_frame_counter = 0
                #         batch_accumulation_time = batch_end_time - batch_start_time
                #         print(
                #             f"Time taken to accumulate frames of batch size {BATCH_SIZE} is "
                #             f"{batch_accumulation_time:>20.9f} seconds"
                #         )
                #         multicam_server_logger.info(
                #             f"Time taken to accumulate frames of batch size {BATCH_SIZE} is "
                #             f"{batch_accumulation_time:>20.9f} seconds"
                #         )
                #         batch_start_time = time.time()
            except Exception as error:
                # if an exception is raised, then release the camera stream and set the flag to False
                print(
                    f'Exception raised while placing the frame in the buffer from {cam.cam_name} '
                    f'(url: {cam.cam_ip})) due to {error}. Releasing the stream...'
                )
                cam.release()
                cam.is_initialized = False
        else:
            # # calculate the end time
            # fps_end_time = time.time()
            # # calculate the time elapsed
            # time_elapsed = fps_end_time - fps_start_time
            # # calculate the frame rate
            # frame_rate = fps_frame_counter / time_elapsed
            # print(
            #     f"Camera stream from {cam.cam_name} (url: {cam.cam_ip})) ended. Frame rate: {frame_rate:.4f} fps"
            # )
            # multicam_server_logger.info(
            #     f"Camera stream from {cam.cam_name} (url: {cam.cam_ip})) ended. Frame rate: {frame_rate:.4f} fps"
            # )

            # destroy the camera object since the camera stream was not initialized
            print(
                f"Camera stream from {cam.cam_name} (url: {cam.cam_ip})) is not accessible. Destroying the camera "
                f"object...")
            del cam
            # put the thread to sleep for 10 seconds
            print(
                f"Putting the thread to sleep for {cam_name} (url: {cam_ip})) for {IP_CAM_REINIT_WAIT_DURATION} "
                f"seconds..."
            )
            time.sleep(IP_CAM_REINIT_WAIT_DURATION)
            # again try to recreate a new camera object
            print(f"Creating a new camera object for {cam_name} (url: {cam_ip}))...")
            cam = IPCamera(cam_name, cam_ip, shared_buffer)
            cameras.append(cam)


def producer_main(shared_buffer):
    # Create a thread for each camera and start the thread
    for cam_name in IP_CAMS:
        cam_ip = IP_CAMS[cam_name]
        cam_thread = threading.Thread(target=create_camera, args=(cam_name, cam_ip, shared_buffer))
        cam_thread.start()