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

# get the screen width and height
screen_w, screen_h = pyautogui.size()

# authenticate with Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # creates local webserver and auto handles authentication
drive = GoogleDrive(gauth)

viewBox = (640, 480)

def draw_circle_with_antialiasing(frame, center, radius, color, thickness):
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, color, thickness=thickness)

    # apply Gaussian blur
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    frame = cv2.addWeighted(frame, 1.0, mask, 0.5, 0)
    return frame

def create_output_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def save_svg_content(folder, svg_content):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'hand_image_{current_time}.svg'
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'w') as f:
        f.write(svg_content)

    # ID of the folder in Google Drive
    folder_id = '1IyhWRWJZ9KPyfJPC9f9xatIs-ac_cyLa'
    #https://drive.google.com/drive/folders/1IyhWRWJZ9KPyfJPC9f9xatIs-ac_cyLa?usp=drive_link
    # upload the image to Google Drive
    file = drive.CreateFile({'parents': [{'id': folder_id}]})
    file.SetContentFile(file_path)
    file.Upload()

def put_custom_text(frame, text, position, font_path, font_size, color):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)

    # load custom font
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame_bgr


def get_fingertip_and_wrist_coordinates(hand_landmarks):
    # check if hand landmarks are valid
    if hand_landmarks is None:
        print("Error: No hand landmarks detected.")
        return None, None, None, None

    # extract wrist coordinates
    wrist = hand_landmarks.landmark[0]
    wrist_coords = (int(wrist.x * screen_w), int(wrist.y * screen_h))

    if hasattr(hand_landmarks, "multi_handedness") and hand_landmarks.multi_handedness:
        handedness = hand_landmarks.multi_handedness[0]

        if handedness.classification[0].label == 'Right':
            # right hand
            fingertips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
        else:
            # left hand
            fingertips = [hand_landmarks.landmark[i] for i in [3, 7, 11, 15, 19]]

        if not fingertips:
            print("Error: No fingertips detected.")
            return None, None, None, None

        fingertip_coords = [(int(fingertip.x * screen_w), int(fingertip.y * screen_h)) for fingertip in fingertips]
    else:
        fingertips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]

        if not fingertips:
            print("Error: No fingertips detected.")
            return None, None, None, None

        fingertip_coords = [(int(fingertip.x * screen_w), int(fingertip.y * screen_h)) for fingertip in fingertips]

    angle, selected_fingertip = calculate_angle(wrist_coords, fingertip_coords, viewBox)

    return wrist_coords, fingertip_coords, angle, selected_fingertip


def calculate_angle(wrist, fingertips, viewBox):
    # randomly select one fingertip
    selected_fingertip = random.choice(fingertips)

    # calculate the angle between the wrist and the selected fingertip
    delta_x = selected_fingertip[0] - wrist[0]
    delta_y = selected_fingertip[1] - wrist[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    scaled_fingertip_x = int((selected_fingertip[0] / screen_w) * viewBox[0])
    scaled_fingertip_y = int((selected_fingertip[1] / screen_h) * viewBox[1])

    return angle, (scaled_fingertip_x, scaled_fingertip_y)

def main():
    output_folder = create_output_folder("hand_images")
    image_index = 0

    # Open the default camera
    #cap = cv2.VideoCapture(0)
    # Open the external camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    countdown_duration = 3
    start_time = None

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

    # list of colors
    colors = {
        "Vermelho": "#D10404",
        "Laranja": "#F94600",
        "Azul Turquesa": "#308A85",
        "Amarelo": "#FFC738",
        "Rosa": "#D7A19A"
    }

    color_names = list(colors.keys())
    selected_color_index = 0

    if selected_color_index < 0 or selected_color_index >= len(color_names):
        print(
            f"Error: selected_color_index {selected_color_index} is out of range. It must be between 0 and {len(color_names) - 1}.")
        exit(1)

    cv2.namedWindow('Camera', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # flip the frame horizontally to correct for inversion when using the pc camera
        #frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (int(screen_w//1.5), int(screen_h//1.5)))

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

        top_left_x = (screen_w - frame.shape[1]) // 2
        top_left_y = (screen_h - frame.shape[0]) // 2

        canvas[top_left_y:top_left_y + frame.shape[0], top_left_x:top_left_x + frame.shape[1]] = frame

        center = (int((screen_w - frame.shape[1]) // 2 + frame.shape[1]//2), int(screen_h * 0.915))
        radius = 75

        top_left_whole = (0, 0)
        bottom_right_whole = (int(screen_w), int(screen_h))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect hands in the frame
        results = hands.process(rgb_frame)

        if start_time is None:
            canvas = put_custom_text(canvas, "Insert your hand in the box", (int((screen_w - frame.shape[1]) // 2), int(screen_h * 0.04)),
                                    "Montserrat-Medium.ttf", 92, (255, 255, 255))
            canvas = put_custom_text(canvas, "to create your own artwork!", (int((screen_w - frame.shape[1]) // 2), int(screen_h * 0.85)),
                                    "Montserrat-Medium.ttf", 91, (255, 255, 255))

        # elapsed time for countdown
        if start_time is not None:
            elapsed_time = time.time() - start_time
            time_remaining = max(0, countdown_duration - int(elapsed_time))

            if elapsed_time < countdown_duration and results.multi_hand_landmarks:
                canvas = draw_circle_with_antialiasing(canvas, center, radius, (255, 255, 255), 2)
                if time_remaining != 1:
                    canvas = put_custom_text(canvas, f" {time_remaining} ", (int((screen_w - frame.shape[1]) // 2 + (frame.shape[1]//2.145)), int(screen_h * 0.867)),
                                            "Montserrat-Medium.ttf", 80, (255, 255, 255))
                elif time_remaining == 1:
                    canvas = put_custom_text(canvas, f" {time_remaining} ", (int((screen_w - frame.shape[1]) // 2 + (frame.shape[1]//2.125)), int(screen_h * 0.867)),
                                            "Montserrat-Medium.ttf", 80, (255, 255, 255))

            if elapsed_time < countdown_duration + saved_image_duration and results.multi_hand_landmarks:
                canvas = put_custom_text(canvas, "Please hold still.", (int((screen_w - frame.shape[1]) // 2 + (frame.shape[1]//5)), int(screen_h * 0.04)),
                                        "Montserrat-Medium.ttf", 92, (255, 255, 255))

            # check if countdown is complete
            if elapsed_time >= countdown_duration and flag_last_message == 0 and flag_saved_image == 0 and flag_done == 0 and results.multi_hand_landmarks:
                flag_saved_image = 1
                saved_image_time = time.time()

            # reset the countdown if the hand disappears during countdown
            if elapsed_time < countdown_duration and not results.multi_hand_landmarks:
                start_time = None

            if saved_image_time is not None:
                if (time.time() - saved_image_time >= saved_image_duration):
                    if (flag_saved_image == 0 and flag_done == 1 and flag_show_done_message == 0):
                        done_time = time.time()
                        flag_show_done_message = 1

            if flag_show_done_message == 1 and done_time is not None:
                if (time.time() - done_time < done_duration):
                    canvas = put_custom_text(canvas, "Done!", (int((screen_w - frame.shape[1]) // 2 + (frame.shape[1]//2.5)), int(screen_h * 0.04)),
                                            "Montserrat-Medium.ttf", 92, (255, 255, 255))
                elif (time.time() - done_time >= done_duration):
                    flag_last_message = 1
                    last_message_time = time.time()
                    flag_show_last_message = 1
                    flag_done = 0
                    flag_show_done_message = 0

            if flag_show_last_message == 1:
                cv2.rectangle(canvas, top_left_whole, bottom_right_whole, (0, 0, 0), thickness=cv2.FILLED)
                if last_message_time is not None:
                    if time.time() - last_message_time < last_message_duration:
                        canvas = put_custom_text(canvas, "Enjoy the exhibition", (int((screen_w - frame.shape[1]) // 2 + frame.shape[1]//7), int(screen_h * 0.15)),
                                                "Montserrat-Medium.ttf", 92, (255, 255, 255))
                        canvas = put_custom_text(canvas, "Leslie David", (int((screen_w - frame.shape[1]) // 2 + frame.shape[1]//5.5), int(screen_h * 0.4)),
                                                "Montserrat-Medium.ttf", 140, (255, 255, 255))
                        canvas = put_custom_text(canvas, "See the results inside", (int((screen_w - frame.shape[1]) // 2 + frame.shape[1]//7.5), int(screen_h * 0.75)),
                                                "Montserrat-Medium.ttf", 92, (255, 255, 255))
                    if time.time() - last_message_time >= last_message_duration:
                        # reset start time and last message time
                        start_time = None
                        last_message_time = None
                        flag_last_message = 0
                        flag_show_last_message = 0

        # process hand landmarks and draw contours
        if results.multi_hand_landmarks:
            if start_time is None:
                start_time = time.time()

            for hand_landmarks in results.multi_hand_landmarks:
                # extract hand region based on landmarks
                hand_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in
                               hand_landmarks.landmark]
                min_x = min(hand_points, key=lambda x: x[0])[0]
                max_x = max(hand_points, key=lambda x: x[0])[0]
                min_y = min(hand_points, key=lambda x: x[1])[1]
                max_y = max(hand_points, key=lambda x: x[1])[1]

                if min_x < 0 or min_y < 0 or max_x >= frame.shape[1] or max_y >= frame.shape[0]:
                    print("Error: Hand region coordinates out of bounds.")
                    continue

                padding = 20
                min_x = max(0, min_x - padding)
                max_x = min(frame.shape[1] - 1, max_x + padding)
                min_y = max(0, min_y - padding)
                max_y = min(frame.shape[0] - 1, max_y + padding)

                hand_region = frame[min_y:max_y, min_x:max_x].copy()

                gray_hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

                _, thresh = cv2.threshold(gray_hand_region, 128, 255, cv2.THRESH_BINARY)

                # find contours in the hand region
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # smooth the contour
                smoothed_contours = [cv2.approxPolyDP(contour, 0.0001 * cv2.arcLength(contour, True), True) for contour
                                     in contours]

                fill_color = colors[color_names[selected_color_index]]

                wrist_coords, fingertip_coords, angle, selected_fingertip = get_fingertip_and_wrist_coordinates(hand_landmarks)

                svg_content = contours_to_svg(smoothed_contours, fill_color=fill_color, contour_color=fill_color,
                                              angle=angle, selected_fingertip=selected_fingertip)

                if (flag_saved_image == 1 and flag_last_message == 0):
                    print("selected_color_index: ", selected_color_index)
                    save_svg_content(output_folder, svg_content)
                    image_index += 1
                    flag_saved_image = 0
                    flag_done = 1
                    if selected_color_index + 1 <= len(color_names) - 1:
                        selected_color_index += 1
                    else:
                        selected_color_index = 0

                break

        cv2.imshow('Camera', canvas)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 27 is the Esc key
            break

    cap.release()
    cv2.destroyAllWindows()

def contours_to_svg(contours, fill_color="#D10404", contour_color="#D10404", angle=0, selected_fingertip=None):
    svg_header = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 480">'
    svg_footer = '</svg>'
    svg_elements = []

    print("selected fingertip: ", selected_fingertip)
    print("angle: ", angle)
    # create custom metadata elements for angle, selected fingertip, and color
    metadata = f'<metadata>Angle: {angle} degrees | Selected Fingertip: {selected_fingertip} | Hand Color: {fill_color}</metadata>'

    for contour in contours:
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        polygon = '<polygon points="'
        for point in points:
            polygon += f"{point[0]},{point[1]} "
        polygon += f'" fill="{fill_color}" stroke="{contour_color}" stroke-width="3" filter="url(#blur)"/>'
        svg_elements.append(polygon)

    # add blur effect
    blur_filter = '<filter id="blur">'
    blur_filter += '<feGaussianBlur in="SourceGraphic" stdDeviation="2"/>'
    blur_filter += '</filter>'

    svg_content = svg_header + metadata + blur_filter + ''.join(svg_elements) + svg_footer

    return svg_content

if __name__ == "__main__":
    main()