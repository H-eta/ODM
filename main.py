'''import cv2
import mediapipe as mp
import numpy as np
import time

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize MediaPipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Countdown duration
    countdown_duration = 3
    start_time = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally to correct for inversion
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(rgb_frame)

        # Process hand landmarks and draw contours
        if results.multi_hand_landmarks:
            if start_time is None:
                start_time = time.time()  # Start the countdown timer

            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand region based on landmarks
                hand_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in hand_landmarks.landmark]
                min_x = min(hand_points, key=lambda x: x[0])[0]
                max_x = max(hand_points, key=lambda x: x[0])[0]
                min_y = min(hand_points, key=lambda x: x[1])[1]
                max_y = max(hand_points, key=lambda x: x[1])[1]

                # Ensure hand region coordinates are within frame bounds
                if min_x < 0 or min_y < 0 or max_x >= frame.shape[1] or max_y >= frame.shape[0]:
                    print("Error: Hand region coordinates out of bounds.")
                    continue

                # Increase the size of the square (ROI)
                padding = 20
                min_x = max(0, min_x - padding)
                max_x = min(frame.shape[1] - 1, max_x + padding)
                min_y = max(0, min_y - padding)
                max_y = min(frame.shape[0] - 1, max_y + padding)

                # Extract hand region from frame
                hand_region = frame[min_y:max_y, min_x:max_x].copy()

                # Convert hand region to grayscale
                gray_hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

                # Apply thresholding to the hand region
                _, thresh = cv2.threshold(gray_hand_region, 128, 255, cv2.THRESH_BINARY)

                # Find contours in the hand region
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Create a blank canvas with transparent background
                canvas = np.zeros((max_y - min_y, max_x - min_x, 4), dtype=np.uint8)

                # Draw contours on the canvas
                cv2.drawContours(canvas, contours, -1, (0, 255, 0, 255), thickness=cv2.FILLED)

                # Smooth the contour
                smoothed_canvas = cv2.GaussianBlur(canvas, (7, 7), 2)

                # Save the hand contour and filling as an image with transparent background
                cv2.imwrite('hand_transparent.png', smoothed_canvas)

                # Exit the loop after processing the first hand detected
                break

        # Calculate elapsed time for countdown
        if start_time is not None:
            elapsed_time = time.time() - start_time
            time_remaining = max(0, countdown_duration - int(elapsed_time))

            # Check if countdown is complete
            if elapsed_time >= countdown_duration:
                break

        # Display the camera frame
        cv2.imshow('Camera', frame)

        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''
'''
import cv2
import mediapipe as mp
import numpy as np
import time
import svgwrite

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize MediaPipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Countdown duration
    countdown_duration = 3
    start_time = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally to correct for inversion
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(rgb_frame)

        # Process hand landmarks and draw contours
        if results.multi_hand_landmarks:
            if start_time is None:
                start_time = time.time()  # Start the countdown timer

            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand region based on landmarks
                hand_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in hand_landmarks.landmark]
                min_x = min(hand_points, key=lambda x: x[0])[0]
                max_x = max(hand_points, key=lambda x: x[0])[0]
                min_y = min(hand_points, key=lambda x: x[1])[1]
                max_y = max(hand_points, key=lambda x: x[1])[1]

                # Ensure hand region coordinates are within frame bounds
                if min_x < 0 or min_y < 0 or max_x >= frame.shape[1] or max_y >= frame.shape[0]:
                    print("Error: Hand region coordinates out of bounds.")
                    continue

                # Increase the size of the square (ROI)
                padding = 20
                min_x = max(0, min_x - padding)
                max_x = min(frame.shape[1] - 1, max_x + padding)
                min_y = max(0, min_y - padding)
                max_y = min(frame.shape[0] - 1, max_y + padding)

                # Extract hand region from frame
                hand_region = frame[min_y:max_y, min_x:max_x].copy()

                # Convert hand region to grayscale
                gray_hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

                # Apply thresholding to the hand region
                _, thresh = cv2.threshold(gray_hand_region, 128, 255, cv2.THRESH_BINARY)

                # Find contours in the hand region
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Create a blank canvas with transparent background
                canvas = np.zeros((max_y - min_y, max_x - min_x, 4), dtype=np.uint8)

                # Draw contours on the canvas
                cv2.drawContours(canvas, contours, -1, (0, 255, 0, 255), thickness=cv2.FILLED)

                # Smooth the contour
                smoothed_canvas = cv2.GaussianBlur(canvas, (7, 7), 2)

                # Convert contour data to SVG format
                svg_content = contours_to_svg(contours)

                # Write SVG content to file
                with open('hand_contour.svg', 'w') as f:
                    f.write(svg_content)

                # Exit the loop after processing the first hand detected
                break

        # Calculate elapsed time for countdown
        if start_time is not None:
            elapsed_time = time.time() - start_time
            time_remaining = max(0, countdown_duration - int(elapsed_time))

            # Check if countdown is complete
            if elapsed_time >= countdown_duration:
                break

        # Display the camera frame
        cv2.imshow('Camera', frame)

        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

def contours_to_svg(contours):
    dwg = svgwrite.Drawing(filename='hand_contour.svg', size=('100%', '100%'), debug=True)
    for contour in contours:
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        dwg.add(dwg.polyline(points, fill='none', stroke='black'))
    return dwg.tostring()

if __name__ == "__main__":
    main()
'''

'''
import cv2
import mediapipe as mp
import numpy as np
import time
import svgwrite

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize MediaPipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Countdown duration
    countdown_duration = 3
    start_time = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally to correct for inversion
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(rgb_frame)

        # Process hand landmarks and draw contours
        if results.multi_hand_landmarks:
            if start_time is None:
                start_time = time.time()  # Start the countdown timer

            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand region based on landmarks
                hand_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in hand_landmarks.landmark]
                min_x = min(hand_points, key=lambda x: x[0])[0]
                max_x = max(hand_points, key=lambda x: x[0])[0]
                min_y = min(hand_points, key=lambda x: x[1])[1]
                max_y = max(hand_points, key=lambda x: x[1])[1]

                # Ensure hand region coordinates are within frame bounds
                if min_x < 0 or min_y < 0 or max_x >= frame.shape[1] or max_y >= frame.shape[0]:
                    print("Error: Hand region coordinates out of bounds.")
                    continue

                # Increase the size of the square (ROI)
                padding = 20
                min_x = max(0, min_x - padding)
                max_x = min(frame.shape[1] - 1, max_x + padding)
                min_y = max(0, min_y - padding)
                max_y = min(frame.shape[0] - 1, max_y + padding)

                # Extract hand region from frame
                hand_region = frame[min_y:max_y, min_x:max_x].copy()

                # Convert hand region to grayscale
                gray_hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

                # Apply thresholding to the hand region
                _, thresh = cv2.threshold(gray_hand_region, 128, 255, cv2.THRESH_BINARY)

                # Find contours in the hand region
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Smooth the contour
                smoothed_contours = [cv2.approxPolyDP(contour, 0.0001 * cv2.arcLength(contour, True), True) for contour in contours]

                # Convert contour data to SVG format
                svg_content = contours_to_svg(smoothed_contours)

                # Write SVG content to file
                with open('hand_contour.svg', 'w') as f:
                    f.write(svg_content)

                # Exit the loop after processing the first hand detected
                break

        # Calculate elapsed time for countdown
        if start_time is not None:
            elapsed_time = time.time() - start_time
            time_remaining = max(0, countdown_duration - int(elapsed_time))

            # Check if countdown is complete
            if elapsed_time >= countdown_duration:
                break

        # Display the camera frame
        cv2.imshow('Camera', frame)

        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

def contours_to_svg(contours):
    dwg = svgwrite.Drawing(filename='hand_contour.svg', size=('100%', '100%'), debug=True)
    for contour in contours:
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        dwg.add(dwg.polyline(points, fill='none', stroke='black'))
    return dwg.tostring()

if __name__ == "__main__":
    main()
'''


import warnings
import os

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from PIL import Image, ImageDraw, ImageFont
import random

import datetime


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Get the screen width and height
screen_w, screen_h = pyautogui.size()

# Authenticate with Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
drive = GoogleDrive(gauth)

viewBox = (640, 480)  # Assuming viewBox dimensions are 640x480

def draw_circle_with_antialiasing(frame, center, radius, color, thickness):
    # Create a mask for the circle
    mask = np.zeros_like(frame)

    # Draw the circle on the mask with a larger thickness
    cv2.circle(mask, center, radius, color, thickness=thickness)

    # Apply Gaussian blur to the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Add the blurred circle to the original frame
    frame = cv2.addWeighted(frame, 1.0, mask, 0.5, 0)

    return frame

def create_output_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

#def save_svg_content(folder, index, svg_content):
#    file_path = os.path.join(folder, f'hand_image_{index}.svg')
#    with open(file_path, 'w') as f:
#        f.write(svg_content)
def save_svg_content(folder, svg_content):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'hand_image_{current_time}.svg'
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'w') as f:
        f.write(svg_content)

    # Specify the ID of the destination folder in Google Drive
    folder_id = '1IyhWRWJZ9KPyfJPC9f9xatIs-ac_cyLa'
    #https://drive.google.com/drive/folders/1IyhWRWJZ9KPyfJPC9f9xatIs-ac_cyLa?usp=drive_link
    # Upload the image to Google Drive
    file = drive.CreateFile({'parents': [{'id': folder_id}]})
    file.SetContentFile(file_path)
    file.Upload()

def put_custom_text(frame, text, position, font_path, font_size, color):
    # Convert the frame to RGB (Pillow uses RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to Pillow Image
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)

    # Load custom font
    font = ImageFont.truetype(font_path, font_size)

    # Add text to image
    draw.text(position, text, font=font, fill=color)

    # Convert back to OpenCV format
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame_bgr



def get_fingertip_and_wrist_coordinates(hand_landmarks):
    # Check if hand landmarks are valid
    if hand_landmarks is None:
        print("Error: No hand landmarks detected.")
        return None, None, None, None

    # Extract wrist coordinates
    wrist = hand_landmarks.landmark[0]  # Assuming the wrist landmark is the first one
    wrist_coords = (int(wrist.x * screen_w), int(wrist.y * screen_h))

    # Check if handedness information is available
    if hasattr(hand_landmarks, "multi_handedness") and hand_landmarks.multi_handedness:
        # Extract handedness information
        handedness = hand_landmarks.multi_handedness[0]  # Assuming only one hand is detected

        # Extract fingertips coordinates based on handedness
        if handedness.classification[0].label == 'Right':
            # Extract fingertip coordinates for the right hand
            fingertips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
        else:  # Left hand
            # Extract fingertip coordinates for the left hand
            fingertips = [hand_landmarks.landmark[i] for i in [3, 7, 11, 15, 19]]

        # Ensure fingertip coordinates are valid
        if not fingertips:
            print("Error: No fingertips detected.")
            return None, None, None, None

        fingertip_coords = [(int(fingertip.x * screen_w), int(fingertip.y * screen_h)) for fingertip in fingertips]
    else:
        # If handedness information is not available, assume it's the right hand
        # Extract fingertip coordinates for the right hand
        fingertips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]

        # Ensure fingertip coordinates are valid
        if not fingertips:
            print("Error: No fingertips detected.")
            return None, None, None, None

        fingertip_coords = [(int(fingertip.x * screen_w), int(fingertip.y * screen_h)) for fingertip in fingertips]

    angle, selected_fingertip = calculate_angle(wrist_coords, fingertip_coords, viewBox)

    return wrist_coords, fingertip_coords, angle, selected_fingertip



def calculate_angle(wrist, fingertips, viewBox):
    # Randomly select one fingertip
    selected_fingertip = random.choice(fingertips)

    # Calculate the angle between the wrist and the selected fingertip
    delta_x = selected_fingertip[0] - wrist[0]
    delta_y = selected_fingertip[1] - wrist[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    scaled_fingertip_x = int((selected_fingertip[0] / screen_w) * viewBox[0])
    scaled_fingertip_y = int((selected_fingertip[1] / screen_h) * viewBox[1])

    return angle, (scaled_fingertip_x, scaled_fingertip_y)  # Return the angle and the selected fingertip coordinates

def main():
    output_folder = create_output_folder("hand_images")
    image_index = 0

    # Open the default camera
    cap = cv2.VideoCapture(0)
    # Open the external camera
    #cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize MediaPipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Countdown duration
    countdown_duration = 3
    start_time = None

    # Last message display duration
    last_message_duration = 8
    last_message_time = None

    flag_last_message = 0
    flag_show_last_message = 0

    flag_saved_image = 0
    saved_image_time = None
    saved_image_duration = 2

    # "Done!" message
    flag_done = 0
    done_time = None
    done_duration = 2
    flag_show_done_message = 0

    # List of colors
    colors = {
        "Vermelho": "#D10404",
        "Laranja": "#F94600",
        "Azul Turquesa": "#308A85",
        "Amarelo": "#FFC738",
        "Rosa": "#D7A19A"
    }

    # Select a color by index
    color_names = list(colors.keys())
    selected_color_index = 3

    # Verify if the selected_color_index is within the range of the colors list
    if selected_color_index < 0 or selected_color_index >= len(color_names):
        print(
            f"Error: selected_color_index {selected_color_index} is out of range. It must be between 0 and {len(color_names) - 1}.")
        exit(1)

    # Create a named window and set it to fullscreen
    cv2.namedWindow('Camera', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally to correct for inversion when using the pc camera
        frame = cv2.flip(frame, 1)
        #frame = cv2.resize(frame, (screen_w/3, screen_h/3))

        # Draw a circle with anti-aliasing
        center = (int(screen_w * 0.161), int(screen_h * 0.225))
        radius = 100

        # Define the top-left and bottom-right corners of the rectangle
        top_left_left = (0, 0)
        bottom_right_left = (int(screen_w * 0.05), int(screen_h))

        top_left_right = (int(screen_w * 0.28), 0)
        bottom_right_right = (int(screen_w), int(screen_h))

        top_left_top = (0, 0)
        bottom_right_top = (int(screen_w), int(screen_h * 0.07))

        top_left_bottom = (0, int(screen_h * 0.38))
        bottom_right_bottom = (int(screen_w), int(screen_h))

        # Draw a rectangle on the frame
        #cv2.rectangle(frame, top_left_left, bottom_right_left, (0, 0, 0), thickness=cv2.FILLED)
        #cv2.rectangle(frame, top_left_right, bottom_right_right, (0, 0, 0), thickness=cv2.FILLED)
        #cv2.rectangle(frame, top_left_top, bottom_right_top, (0, 0, 0), thickness=cv2.FILLED)
        #cv2.rectangle(frame, top_left_bottom, bottom_right_bottom, (0, 0, 0), thickness=cv2.FILLED)

        top_left_whole = (0, 0)
        bottom_right_whole = (int(screen_w), int(screen_h))

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(rgb_frame)

        # Display instruction text before any hand is detected
        if start_time is None:
            frame = put_custom_text(frame, "Insert your hand in the box", (int(screen_w * 0.049), int(screen_h * 0.02)),
                                    "Montserrat-Medium.ttf", 32, (255, 255, 255))
            frame = put_custom_text(frame, "to create your own artwork!", (int(screen_w * 0.048), int(screen_h * 0.395)),
                                    "Montserrat-Medium.ttf", 32, (255, 255, 255))
            #cv2.putText(frame, "Insert your hand in the box", (int(screen_w * 0.05), int(screen_h * 0.042)),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #cv2.putText(frame, "to create your own artwork!", (int(screen_w * 0.05), int(screen_h * 0.42)),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Calculate elapsed time for countdown
        if start_time is not None:
            elapsed_time = time.time() - start_time
            time_remaining = max(0, countdown_duration - int(elapsed_time))

            # Display countdown
            if elapsed_time < countdown_duration and results.multi_hand_landmarks:
                #if time_remaining > 1:
                #    cv2.putText(frame, f"Count down: {time_remaining} seconds", (int(screen_w * 0.07), int(screen_h * 0.08)),
                #                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #elif time_remaining == 1:
                #    cv2.putText(frame, f"Count down: {time_remaining} second", (int(screen_w * 0.07), int(screen_h * 0.08)),
                #                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.circle(frame, center=(int(screen_w * 0.161), int(screen_h * 0.22)), radius=100, color=(255, 255, 255),
                #           thickness=2)
                frame = draw_circle_with_antialiasing(frame, center, radius, (255, 255, 255), 2)
                if time_remaining != 1:
                    frame = put_custom_text(frame, f" {time_remaining} ", (int(screen_w * 0.14), int(screen_h * 0.18)),
                                            "Montserrat-Medium.ttf", 80, (255, 255, 255))
                elif time_remaining == 1:
                    frame = put_custom_text(frame, f" {time_remaining} ", (int(screen_w * 0.142), int(screen_h * 0.18)),
                                            "Montserrat-Medium.ttf", 80, (255, 255, 255))
                #cv2.putText(frame, f" {time_remaining} ", (int(screen_w * 0.12), int(screen_h * 0.25)),
                #            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)

            if elapsed_time < countdown_duration + saved_image_duration and results.multi_hand_landmarks:
                frame = put_custom_text(frame, "Please hold still.", (int(screen_w * 0.1), int(screen_h * 0.02)),
                                        "Montserrat-Medium.ttf", 32, (255, 255, 255))

            # Check if countdown is complete
            if elapsed_time >= countdown_duration and flag_last_message == 0 and flag_saved_image == 0 and flag_done == 0 and results.multi_hand_landmarks:
                #done_time = time.time()
                #flag_done = 1
                flag_saved_image = 1
                saved_image_time = time.time()

            # Reset the countdown if the hand disappears during countdown
            if elapsed_time < countdown_duration and not results.multi_hand_landmarks:
                start_time = None

            if saved_image_time is not None:
                if (time.time() - saved_image_time >= saved_image_duration):
                    if (flag_saved_image == 0 and flag_done == 1 and flag_show_done_message == 0):
                        done_time = time.time()
                        flag_show_done_message = 1
                        #last_message_time = time.time()
                        #flag_show_last_message = 1

            if flag_show_done_message == 1 and done_time is not None:
                if (time.time() - done_time < done_duration):
                    frame = put_custom_text(frame, "Done!", (int(screen_w * 0.14), int(screen_h * 0.02)),
                                            "Montserrat-Medium.ttf", 32, (255, 255, 255))
                elif (time.time() - done_time >= done_duration):
                    flag_last_message = 1
                    last_message_time = time.time()
                    flag_show_last_message = 1
                    flag_done = 0
                    flag_show_done_message = 0

            if flag_show_last_message == 1:
                cv2.rectangle(frame, top_left_whole, bottom_right_whole, (0, 0, 0), thickness=cv2.FILLED)
                # Display last message for specified duration
                if last_message_time is not None:
                    #print(time.time() - last_message_time)
                    if time.time() - last_message_time < last_message_duration:
                        frame = put_custom_text(frame, "Enjoy the exhibition", (int(screen_w * 0.085), int(screen_h * 0.1)),
                                                "Montserrat-Medium.ttf", 32, (255, 255, 255))
                        frame = put_custom_text(frame, "Leslie David", (int(screen_w * 0.0818), int(screen_h * 0.17)),
                                                "Montserrat-Medium.ttf", 55, (255, 255, 255))
                        frame = put_custom_text(frame, "See the results inside", (int(screen_w * 0.078), int(screen_h * 0.35)),
                                                "Montserrat-Medium.ttf", 32, (255, 255, 255))
                    if time.time() - last_message_time >= last_message_duration:
                        # Reset start time and last message time
                        start_time = None
                        last_message_time = None
                        flag_last_message = 0
                        flag_show_last_message = 0

        # Process hand landmarks and draw contours
        if results.multi_hand_landmarks:
            if start_time is None:
                start_time = time.time()  # Start the countdown timer

            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand region based on landmarks
                hand_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in
                               hand_landmarks.landmark]
                min_x = min(hand_points, key=lambda x: x[0])[0]
                max_x = max(hand_points, key=lambda x: x[0])[0]
                min_y = min(hand_points, key=lambda x: x[1])[1]
                max_y = max(hand_points, key=lambda x: x[1])[1]

                # Ensure hand region coordinates are within frame bounds
                if min_x < 0 or min_y < 0 or max_x >= frame.shape[1] or max_y >= frame.shape[0]:
                    print("Error: Hand region coordinates out of bounds.")
                    continue

                # Increase the size of the square
                padding = 20
                min_x = max(0, min_x - padding)
                max_x = min(frame.shape[1] - 1, max_x + padding)
                min_y = max(0, min_y - padding)
                max_y = min(frame.shape[0] - 1, max_y + padding)

                # Extract hand region from frame
                hand_region = frame[min_y:max_y, min_x:max_x].copy()

                # Convert hand region to grayscale
                gray_hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

                # Apply thresholding to the hand region
                _, thresh = cv2.threshold(gray_hand_region, 128, 255, cv2.THRESH_BINARY)

                # Find contours in the hand region
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Smooth the contour
                smoothed_contours = [cv2.approxPolyDP(contour, 0.0001 * cv2.arcLength(contour, True), True) for contour
                                     in contours]

                # Convert contour data to SVG format
                fill_color = colors[color_names[selected_color_index]]
                #svg_content = contours_to_svg(smoothed_contours, fill_color=fill_color, contour_color=fill_color)




                wrist_coords, fingertip_coords, angle, selected_fingertip = get_fingertip_and_wrist_coordinates(hand_landmarks)
                #angle, selected_fingertip = calculate_angle(wrist_coords, fingertip_coords, viewBox)


                # Convert angle to a single value if needed (assuming calculate_angle returns an array)
                #angle = angle[0] if isinstance(angle, np.ndarray) else angle
                svg_content = contours_to_svg(smoothed_contours, fill_color=fill_color, contour_color=fill_color,
                                              angle=angle, selected_fingertip=selected_fingertip)




                # Write SVG content to file
                #with open('hand_contour_fill.svg', 'w') as f:
                #    f.write(svg_content)
                # Save SVG content to file with indexing
                if (flag_saved_image == 1 and flag_last_message == 0):
                    print("selected_color_index: ", selected_color_index)
                    #save_svg_content(output_folder, image_index, svg_content)
                    save_svg_content(output_folder, svg_content)
                    image_index += 1
                    flag_saved_image = 0
                    #flag_last_message = 1
                    flag_done = 1
                    if selected_color_index + 1 <= len(color_names) - 1:
                        selected_color_index += 1
                    else:
                        selected_color_index = 0

                # Exit the loop after processing the first hand detected
                break

        # Display the camera frame
        cv2.imshow('Camera', frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 27 is the Esc key
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()





def contours_to_svg(contours, fill_color="#D10404", contour_color="#D10404", angle=0, selected_fingertip=None):
    svg_header = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 480">'
    svg_footer = '</svg>'
    svg_elements = []

    print("selected fingertip: ", selected_fingertip)
    print("angle: ", angle)
    # Create custom metadata elements for angle and selected fingertip
    metadata = f'<metadata>Angle: {angle} degrees | Selected Fingertip: {selected_fingertip} | Hand Color: {fill_color}</metadata>'

    for contour in contours:
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        polygon = '<polygon points="'
        for point in points:
            polygon += f"{point[0]},{point[1]} "
        polygon += f'" fill="{fill_color}" stroke="{contour_color}" stroke-width="3" filter="url(#blur)"/>'
        svg_elements.append(polygon)

    # Add filter element for blur effect
    blur_filter = '<filter id="blur">'
    blur_filter += '<feGaussianBlur in="SourceGraphic" stdDeviation="2"/>'
    blur_filter += '</filter>'

    svg_content = svg_header + metadata + blur_filter + ''.join(svg_elements) + svg_footer

    return svg_content

if __name__ == "__main__":
    main()