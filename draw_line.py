import cv2

# List to store the points of the line
line_points = []

# Target size for resizing
target_width, target_height = 640, 360


# Mouse callback function
def draw_line(event, x, y, flags, param):
    global line_points, img_copy

    # If left button is clicked, store the point
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:  # Only allow two points
            line_points.append((x, y))
            print(f"Point selected: {x}, {y}")

            # Draw the points on the image
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)

            # Draw the line if two points are selected
            if len(line_points) == 2:
                cv2.line(img_copy, line_points[0], line_points[1], (0, 0, 255), 2)
                cv2.imshow("Frame", img_copy)


# Open a frame (can be from a video or image)
frame_path = "Screenshot 2024-09-12 144926.png"  # Replace with your image or frame path

# Read the frame
img = cv2.imread(frame_path)
if img is None:
    print(f"Error: Unable to load the image from {frame_path}")
    exit()

# Resize the image to 1920x1080
img = cv2.resize(img, (target_width, target_height))

# Create a copy of the resized image to display the changes
img_copy = img.copy()

# Create a window and set a mouse callback to detect mouse clicks
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_line)

# Display the frame and wait for the user to draw the line
while True:
    cv2.imshow("Frame", img_copy)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the final line points
if len(line_points) == 2:
    print(f"Line drawn from {line_points[0]} to {line_points[1]}")
else:
    print("Line not fully drawn. Please select two points.")

# Clean up
cv2.destroyAllWindows()
