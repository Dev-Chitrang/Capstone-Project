import cv2
import pyttsx3
from ultralytics import YOLO

# Load YOLO model
yolo = YOLO('yolov8s.pt')

# Load video capture
videoCap = cv2.VideoCapture(0)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed

# Real-world calibration parameters
KNOWN_DISTANCE = 100  # cm (Set a known distance for calibration)
KNOWN_HEIGHT = 165  # cm (Average human height)
FOCAL_LENGTH = 700  # Adjusted through real-world testing

# Dictionary to track last spoken distances for objects
last_alerts = {}


def getColours(cls_num):
    """Get colors for different classes."""
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)


def calculate_distance(pixel_height):
    """Calculate object distance using the focal length formula."""
    if pixel_height == 0:
        return float('inf')  # Avoid division by zero
    return (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height


def get_direction(x1, frame_width):
    """Determine object direction (Left, Center, Right) based on position."""
    left_threshold = frame_width // 3
    right_threshold = 2 * (frame_width // 3)

    if x1 < left_threshold:
        return "left"
    elif x1 > right_threshold:
        return "right"
    else:
        return "center"


def speak_alert(object_id, distance, class_name, direction):
    """Speak an alert and provide navigation guidance."""
    global last_alerts

    if distance > 500:  # Ignore distant objects (speak only once)
        if object_id not in last_alerts:
            message = f"{class_name} detected at {distance:.1f} centimeters, {direction}."
            engine.say(message)
            engine.runAndWait()
            last_alerts[object_id] = distance
        return

    # Guidance based on object position
    guidance = ""
    if distance < 100:  # If object is very close
        if direction == "left":
            guidance = "Move slightly to the right."
        elif direction == "right":
            guidance = "Move slightly to the left."
        elif direction == "center":
            guidance = "Stop! Adjust left or right."

    # Speak only if distance changes significantly
    last_distance = last_alerts.get(object_id, None)
    if last_distance is None or abs(last_distance - distance) > 10:
        message = f"⚠ Warning! {class_name} at {distance:.1f} cm, {direction}. {guidance}"
        engine.say(message)
        engine.runAndWait()
        last_alerts[object_id] = distance


while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    frame_width = frame.shape[1]  # Get the width of the frame

    results = yolo.track(frame, stream=True)

    for result in results:
        classes_names = result.names  # Get class names

        for i, box in enumerate(result.boxes):
            if box.conf[0] > 0.5:  # Confidence threshold
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                class_name = classes_names[cls]

                # Calculate height of the detected object in pixels
                pixel_height = y2 - y1

                # Compute distance
                distance_cm = calculate_distance(pixel_height)

                # Get color
                colour = getColours(cls)

                # Determine direction
                direction = get_direction(x1, frame_width)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # Display class, confidence & distance
                text = f"{class_name} {box.conf[0]:.2f} | {distance_cm:.1f} cm | {direction}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

                # Speak distance-based alert only when necessary
                speak_alert(i, distance_cm, class_name, direction)

                # Display visual warning if very close
                if distance_cm < 50:
                    cv2.putText(frame, "⚠ DANGER: TOO CLOSE!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show output frame
    cv2.imshow('Blind Navigation Assistance', frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
videoCap.release()
cv2.destroyAllWindows()
