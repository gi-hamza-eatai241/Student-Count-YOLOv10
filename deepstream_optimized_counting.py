import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds

PGIE_CLASS_ID_PERSON = 0
MAX_BATCH_SIZE = 30  # Adjust this based on your hardware capabilities

# Example line coordinates for line-crossing logic (replace with actual values)
line_coordinates = [(253, 168), (296, 358)]

# List of RTSP stream URLs
RTSP_STREAMS = [
    "rtsp://hamzaziizzz:23082001@192.168.1.4:554/stream1"
]


def is_crossing_line(p, line_point_1, line_point_2):
    """
    Helper function to check if a point `p` has crossed the line defined by `line_point_1` and `line_point_2`.
    """
    d1 = (line_point_2[1] - line_point_1[1]) * p[0] - (line_point_2[0] - line_point_1[0]) * p[1] + \
         line_point_2[0] * line_point_1[1] - line_point_2[1] * line_point_1[0]
    return d1 < 0


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    This function is called for each frame passed to the OSD. It performs line-crossing logic and updates counters.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == PGIE_CLASS_ID_PERSON:
                bbox_center_x = int((obj_meta.rect_params.left + obj_meta.rect_params.width / 2))
                bbox_center_y = int((obj_meta.rect_params.top + obj_meta.rect_params.height / 2))

                if is_crossing_line((bbox_center_x, bbox_center_y), line_coordinates[0], line_coordinates[1]):
                    print(f"Person {obj_meta.object_id} crossed the line!")

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def create_source_bin(index, uri):
    """
    Create a source bin for each RTSP stream. This allows dynamic handling of multiple streams.
    """
    bin_name = "source-bin-%02d" % index
    bin = Gst.Bin.new(bin_name)

    uridecodebin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uridecodebin:
        print(f"Failed to create uridecodebin for {uri}")
        return None
    uridecodebin.set_property("uri", uri)
    uridecodebin.connect("pad-added", cb_newpad, bin)

    bin.add(uridecodebin)

    return bin

def cb_newpad(decodebin, pad, bin):
    """
    Callback function to link the decoded pad of each stream to the rest of the pipeline.
    """
    caps = pad.get_current_caps()
    structure = caps.get_structure(0)
    caps_string = structure.get_name()
    features = caps.get_features(0)

    # Print caps information for debugging
    print(f"New pad {pad.get_name()} with caps {caps_string}")

    # Check if the pad is video-related, we are only interested in video/x-raw or video/x-raw(memory:NVMM)
    if caps_string.startswith("video/x-raw"):
        if features.contains("memory:NVMM"):
            # If the pad has NVMM memory, we can link it directly
            sink_pad = bin.get_static_pad("src")
            if not sink_pad:
                bin.add_pad(Gst.GhostPad.new("src", pad))
                print("Linked pad to the source bin.")
        else:
            print("Decodebin output does not contain memory:NVMM. Adding nvvideoconvert.")

            # Create nvvideoconvert element
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
            if not nvvidconv:
                print("Failed to create nvvideoconvert")
                return

            bin.add(nvvidconv)
            nvvidconv.sync_state_with_parent()  # Sync state of the new element with the pipeline

            # Set a caps filter for the nvvidconv element to ensure format negotiation
            capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
            if not capsfilter:
                print("Failed to create capsfilter")
                return
            capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)NV12"))

            bin.add(capsfilter)
            capsfilter.sync_state_with_parent()  # Sync state of the capsfilter

            # Link the decodebin pad to nvvidconv sink pad
            if pad.link(nvvidconv.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                print("Failed to link decodebin pad to nvvideoconvert")
                return
            else:
                print("Linked decodebin pad to nvvideoconvert successfully.")

            # Now link the nvvidconv output to the caps filter
            if nvvidconv.link(capsfilter) != Gst.PadLinkReturn.OK:
                print("Failed to link nvvideoconvert to capsfilter")
                return
            else:
                print("Linked nvvideoconvert to capsfilter successfully.")

            # Create a ghost pad to link capsfilter to the rest of the pipeline
            sink_pad = bin.get_static_pad("src")
            if not sink_pad:
                bin.add_pad(Gst.GhostPad.new("src", capsfilter.get_static_pad("src")))
    else:
        print(f"Ignoring pad with caps {caps_string}")


def main():
    # Initialize GStreamer and create the pipeline
    Gst.init(None)

    # Create GStreamer pipeline
    pipeline = Gst.Pipeline()

    # Create nvstreammux to mux the streams
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    if not streammux:
        print("Unable to create nvstreammux")
        sys.exit(1)
    pipeline.add(streammux)

    # Create nvinfer for inference (using YOLO TensorRT model)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        print("Unable to create nvinfer")
        sys.exit(1)
    pgie.set_property('config-file-path', "yolov8x_config.txt")  # Update with your YOLO TensorRT model path
    pipeline.add(pgie)

    # Create nvosd for on-screen display
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")
    if not nvosd:
        print("Unable to create nvosd")
        sys.exit(1)
    pipeline.add(nvosd)

    # Create sink to display the output
    sink = Gst.ElementFactory.make("nveglglessink", "nveglglessink")
    if not sink:
        print("Unable to create sink")
        sys.exit(1)
    pipeline.add(sink)

    # Set streammux properties
    streammux.set_property("batch-size", min(len(RTSP_STREAMS), MAX_BATCH_SIZE))  # Adjust batch size dynamically
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batched-push-timeout", 40000)

    # Add and link the elements for each RTSP stream
    for i, rtsp_url in enumerate(RTSP_STREAMS):
        source_bin = create_source_bin(i, rtsp_url)
        if not source_bin:
            print(f"Failed to create source bin for {rtsp_url}")
            sys.exit(1)
        pipeline.add(source_bin)

        padname = f"sink_{i}"
        # Replace deprecated get_request_pad with request_pad
        sinkpad = streammux.request_pad(streammux.get_pad_template("sink_%u"), padname, None)
        if not sinkpad:
            print(f"Unable to get sink pad for stream {i}")
            sys.exit(1)

        # Wait for a pad to be available for the source bin
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            print(f"Warning: Source pad not available yet for stream {i}. Waiting for pad to be added.")
        else:
            print(f"Source pad found for stream {i}. Linking...")
            result = srcpad.link(sinkpad)
            if result != Gst.PadLinkReturn.OK:
                print(f"Failed to link source pad and sink pad for stream {i}, error: {result}")
                sys.exit(1)

    streammux.link(pgie)
    pgie.link(nvosd)
    nvosd.link(sink)

    # Add probe to osd sink pad for line-crossing logic
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        print("Unable to get sink pad of OSD")
    else:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Run the main loop
    try:
        loop = GLib.MainLoop()
        loop.run()
    except:
        pass

    # Cleanup after execution
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
