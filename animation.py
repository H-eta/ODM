import os
import subprocess
import tkinter as tk
from PIL import Image, ImageTk, ImageOps, ImageFilter
import xml.etree.ElementTree as ET
import pyautogui
import random
import datetime
import time
import threading
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import math

# Path to the folder containing animation frames
ANIMATION_FRAMES_PATH = 'teste2_color_changing/'
SAVED_FRAMES_PATH = 'saved_animation_frames/'
hand_images_folder = 'hand_images_from_drive/'
folder_id = '1IyhWRWJZ9KPyfJPC9f9xatIs-ac_cyLa'
local_folder = 'hand_images_from_drive'
interval = 1  # Check every 1 second

width_, height_ = pyautogui.size()

# Create the directory for saved frames if it doesn't exist
os.makedirs(SAVED_FRAMES_PATH, exist_ok=True)

# Initialize Tkinter
root = tk.Tk()
root.attributes('-fullscreen', True)
canvas = tk.Canvas(root, width=width_, height=height_, bg='#000000')
canvas.pack()

state_dict = {}
new_width = 0
new_height = 0

flag_fade_out = 0

# Authenticate with Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
drive = GoogleDrive(gauth)

# Example Usage
num_rows = 3
num_columns = 5

# Calculate the width and height of each space
space_width = width_ // num_columns
space_height = height_ // num_rows

# Create a list to hold the coordinates of each space
space_coords = []


# Load animation frames with specified hand color
def load_animation_frames(frames_path, hand_color):
    frames = []
    for file_name in sorted(os.listdir(frames_path)):
        if file_name.endswith('.png'):
            frame_path = os.path.join(frames_path, file_name)
            frame = Image.open(frame_path).convert("RGBA")

            # Resize the frame to half its size
            frame = frame.resize((frame.width // 4, frame.height // 4), Image.Resampling.BICUBIC)

            # Colorize the frame if hand color is provided
            if hand_color:
                # Ensure hand_color has the correct format
                if len(hand_color) == 4:
                    # Convert shorthand format (#RGB) to full format (#RRGGBB)
                    hand_color = '#' + ''.join([c * 2 for c in hand_color[1:]])

                # Create a mask based on non-transparent pixels
                alpha = frame.split()[-1]
                mask = Image.new("L", frame.size, 0)
                mask.paste(alpha, alpha)

                # Convert the hand color to RGBA format
                rgba_color = tuple(int(hand_color[i:i + 2], 16) for i in (1, 3, 5)) + (255,)

                # Apply the hand color using the mask
                colored_area = Image.new("RGBA", frame.size, rgba_color)
                frame = Image.composite(colored_area, frame, mask)

            # Apply Gaussian blur filter to the frame
            blurred_frame = frame.filter(ImageFilter.GaussianBlur(radius=3))

            # Append the blurred frame to the list
            frames.append(blurred_frame)

    return frames


# Convert SVG to PNG and display on the same window with animation frames
def display_animation_with_svg(svg_file, animation_frames_path, angle, selected_fingertip, hand_color):
    # Generate a unique folder name based on current time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f'saved_animation_frames_{timestamp}'
    saved_frames_path = os.path.join(SAVED_FRAMES_PATH, folder_name)
    os.makedirs(saved_frames_path, exist_ok=True)

    # Convert SVG to PNG
    #output_png_file = 'hand_images_png/hand_image_0_0_0.png'
    output_png_file = f'hand_images_png/hand_image_{timestamp}.png'
    #output_png_file = os.path.join(saved_frames_path, 'hand_image_0_0_0.png')
    inkscape_path = "D:\\inkscape\\bin\\inkscape.exe"  # Adjust the path as needed
    command = f'"{inkscape_path}" --export-type=png --export-filename="{output_png_file}" --export-area-page "{svg_file}"'
    subprocess.run(command, shell=True)

    # Load the converted PNG image
    png_image = Image.open(output_png_file)
    png_image = png_image.convert("RGBA")

    # Resize the hand image
    original_size = png_image.size
    new_size = (original_size[0], original_size[1])
    png_image = png_image.resize(new_size, Image.Resampling.BICUBIC)
    bbox = png_image.getbbox()
    cropped_png_image = png_image.crop(bbox)

    # Load and rotate the animation frames to the indicated angle
    #rotation_angle = angle  # Rotate clockwise
    #rotation_angle = adjust_angle(angle)
    #rotation_angle = random.randint(20, 160)
    #rotation_angle = 150
    if (-180 <= angle <= -90):
        rotation_angle = angle + 180 + 270
    elif (-90 < angle <= 0):
        rotation_angle = -angle

    #animation_frames_path = "animations/animation1"
    animation_frames = load_animation_frames(animation_frames_path, hand_color)
    #rotated_animation_frames = [frame.rotate(rotation_angle, expand=True) for frame in animation_frames]

    # Get the actual position of the hand image on the canvas
    hand_image_position = (int(width_ // 2 - png_image.width // 4), int(height_ // 2))
    #hand_image_position = (0, 0)  # Place the hand image at (0, 0)

    # Calculate the offset between the original fingertip coordinates and the hand image position
    offset_x = selected_fingertip[0] - hand_image_position[0]
    offset_y = selected_fingertip[1] - hand_image_position[1]

    print("offset_x: ", offset_x)
    print("offset_y: ", offset_y)
    print("angle: ", rotation_angle)

    # Get the bounding box of the final frame
    final_frame_bbox = animation_frames[-1].getbbox()

    print("frame box: ", final_frame_bbox)

    #rotation_angle = 20
    #rotation_angle = 60
    #rotation_angle = 290
    #rotation_angle = 340

    # Combine each animation frame with the hand image and save the result
    for i, frame in enumerate(animation_frames):
        combined_image = Image.new("RGBA", (width_, height_), (0, 0, 0, 0))
        cropped_frame_image = frame.crop(final_frame_bbox)

        #frame_center_x = int(frame_center_x)
        #frame_center_y = int(frame_center_y)

        if (0 <= rotation_angle <= 45):
            #if (animation_frames_path == "animations/animation2"):
                #frame_center_x = hand_image_position[0] - int((final_frame_bbox[2] - final_frame_bbox[0])//1.5) #+ (bbox[2] - bbox[0])//2 # - (final_frame_bbox[2] - final_frame_bbox[0]) #cropped_frame_image.width
                #frame_center_y = hand_image_position[1] - (final_frame_bbox[3] - final_frame_bbox[1])//2 #- cropped_frame_image.height//4
            #else:
            #frame_center_x = hand_image_position[0] - (final_frame_bbox[2] - final_frame_bbox[0]) // 2  # + (bbox[2] - bbox[0])//2 # - (final_frame_bbox[2] - final_frame_bbox[0]) #cropped_frame_image.width
            frame_center_y = hand_image_position[1] - (final_frame_bbox[3] - final_frame_bbox[1]) // 2  # - cropped_frame_image.height//4
            frame_center_x = int(selected_fingertip[0] + offset_x - (final_frame_bbox[2] - final_frame_bbox[0]) // 2)
            #frame_center_y = int(selected_fingertip[1] + offset_y - (final_frame_bbox[3] - final_frame_bbox[1]) // 4)
        elif (45 < rotation_angle <= 90):
            #frame_center_x = hand_image_position[0] + (bbox[2] - bbox[0])//4 - (final_frame_bbox[2] - final_frame_bbox[0])
            #if (animation_frames_path == "animations/animation3"):
            #    frame_center_x = hand_image_position[0] - int(1.5*(final_frame_bbox[2] - final_frame_bbox[0]))
            #    frame_center_y = hand_image_position[1] + int((bbox[3] - bbox[1])//2.5) - (final_frame_bbox[3] - final_frame_bbox[1])//2
            #else:
            frame_center_x = hand_image_position[0] - (final_frame_bbox[2] - final_frame_bbox[0])
            frame_center_y = hand_image_position[1] + int((bbox[3] - bbox[1]) // 2.5) - (final_frame_bbox[3] - final_frame_bbox[1]) // 2
            #frame_center_x = int(selected_fingertip[0] + offset_x - (final_frame_bbox[2] - final_frame_bbox[0]) // 2)
            #frame_center_y = int(selected_fingertip[1] + offset_y)
        elif (270 <= rotation_angle <= 315):
            #if (animation_frames_path == "animations/animation3"):
            #    frame_center_x = hand_image_position[0] + (final_frame_bbox[2] - final_frame_bbox[0])//2 #- hand_image_position[0] // 2
            #    frame_center_y = hand_image_position[1] + int((bbox[3] - bbox[1])//2) - (final_frame_bbox[3] - final_frame_bbox[1])//2
            #else:
            frame_center_x = hand_image_position[0] + (final_frame_bbox[2] - final_frame_bbox[0]) // 4  # - hand_image_position[0] // 2
            frame_center_y = hand_image_position[1] + (bbox[3] - bbox[1]) // 3 - (final_frame_bbox[3] - final_frame_bbox[1]) // 2
            #frame_center_x = int(selected_fingertip[0] + offset_x)
            #frame_center_y = int(selected_fingertip[1] + offset_y)
        elif (315 < rotation_angle <= 360):
            frame_center_x = hand_image_position[0] + (bbox[2] - bbox[0])//8 #- hand_image_position[0] // 2
            frame_center_y = hand_image_position[1] - (final_frame_bbox[3] - final_frame_bbox[1])//2
            #frame_center_x = int(selected_fingertip[0] + offset_x)
            #frame_center_y = int(selected_fingertip[1] + offset_y - (final_frame_bbox[3] - final_frame_bbox[1]) // 4)


        # Rotate the frame
        rotated_frame_image = cropped_frame_image.rotate(rotation_angle, expand=True)

        print(f"Frame {i}: frame_center_x = {frame_center_x}, frame_center_y = {frame_center_y}, angle = {rotation_angle}")

        #rotated_frame_image = cropped_frame_image.rotate(rotation_angle, expand=True)
        combined_image.paste(rotated_frame_image, (frame_center_x, frame_center_y), rotated_frame_image)

        combined_image.paste(cropped_png_image, hand_image_position, cropped_png_image)
        # Apply blur to the combined image
        blurred_combined_image = combined_image.filter(
            ImageFilter.GaussianBlur(radius=3))  # Adjust the radius as needed
        # Save the blurred combined image
        blurred_combined_image.save(os.path.join(saved_frames_path, f'frame_{i:04d}.png'))


# Function to display the saved animation frames in a window
def display_saved_animation(subfolder_path, canvas, x_coord, y_coord, new_width=None, new_height=None, flag_fade_out = 0):

    # Load frames from the subfolder
    frames = []
    for file_name in sorted(os.listdir(subfolder_path)):
        if file_name.endswith('.png'):
            frame_path = os.path.join(subfolder_path, file_name)
            frame = Image.open(frame_path)
            # Resize the frame if new dimensions are provided
            if new_width and new_height:
                frame = frame.resize((new_width, new_height), Image.Resampling.BICUBIC)

            frames.append(frame)

    # Extract the alpha channel from the first frame
    alpha_mask = frames[0].split()[-1]

    # Display frames on the canvas with fading in effect
    for alpha in range(0, 256, 8):
        # Create a faded frame using the alpha channel mask
        faded_frame = frames[0].copy()
        faded_frame.putalpha(alpha_mask.point(lambda p: p * alpha / 255))
        faded_photo = ImageTk.PhotoImage(faded_frame)
        canvas.create_image(x_coord, y_coord, anchor=tk.NW, image=faded_photo)
        canvas.update()
        time.sleep(0.1)

        # Display frames on the canvas with forward direction
    direction = 1
    frame_index = 0
    if flag_fade_out == 0:
        while 0 <= frame_index < len(frames):
            # Convert PIL image to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(frames[frame_index])

            # Display the image on canvas at random coordinates
            canvas_image = canvas.create_image(x_coord, y_coord, anchor=tk.NW, image=photo)

            # Update the canvas
            canvas.update()

            # Delay between frames (adjust as needed)
            time.sleep(0.1)

            # Remove the image from the canvas
            #canvas.delete(canvas_image)

            # Update frame index based on direction
            frame_index += direction

            # Check if the animation should reverse direction
            if frame_index == len(frames) - 1:
                direction = -1
            #elif frame_index == 0:
            #    direction = 1
            elif direction == -1 and frame_index == 0 and flag_fade_out == 0:
                flag_fade_out = 1

    if flag_fade_out == 1:
        canvas.delete(canvas_image)
        # Store a copy of the last frame before starting fade-out
        last_frame_copy = frames[-1].copy()

        # Erase the image from the canvas with fading out effect
        for alpha in range(255, -1, -8):
            #print("alpha: ", alpha)
            #print("fadeout: ", flag_fade_out)
            # Create a faded frame using the alpha channel mask
            faded_frame = last_frame_copy.copy()  # Use the last frame for fade-out
            faded_frame.putalpha(alpha_mask.point(lambda p: p * alpha / 255))
            faded_photo = ImageTk.PhotoImage(faded_frame)
            fadeout_image = canvas.create_image(x_coord, y_coord, anchor=tk.NW, image=faded_photo)
            canvas.update()
            time.sleep(0.1)

    # Remove the image from the canvas after fade-out
    canvas.delete(fadeout_image)

    # Update the canvas after each subfolder animation
    canvas.update()


# Extract metadata from SVG file
def extract_metadata(svg_file):
    metadata = {}
    tree = ET.parse(svg_file)
    root = tree.getroot()
    metadata_tag = root.find('{http://www.w3.org/2000/svg}metadata')
    if metadata_tag is not None:
        metadata_text = metadata_tag.text.strip()
        if metadata_text:
            parts = metadata_text.split('|')
            for part in parts:
                key, value = part.strip().split(':')
                metadata[key.strip()] = value.strip()
    return metadata


# Adjust angle if it's negative and corresponds to tilt direction
def adjust_angle(angle):
    if angle < -90:
        return 180 + angle
    else:
        return angle

def normalize_angle(angle):
    normalized_angle = angle % 360  # Normalize angle between 0 and 360
    if normalized_angle > 180:
        normalized_angle = 360 - normalized_angle
    return normalized_angle



def calculate_grid_coordinates(screen_width, screen_height, rows, cols):
    cell_width = screen_width // cols
    cell_height = screen_height // rows
    coordinates = []
    for row in range(rows):
        for col in range(cols):
            x = col * cell_width
            y = row * cell_height
            coordinates.append((x, y))
    return coordinates


# Function to continuously fetch SVG files
def fetch_svg_files():
    hand_images_folder = 'hand_images_from_drive/'
    processed_files = set()  # Set to keep track of processed SVG files
    animation_folders = ['animations/animation1', 'animations/animation2', 'animations/animation3']
    #animation_folders_list = list(animation_folders.keys())
    while True:
        for svg_file in os.listdir(hand_images_folder):
            if svg_file not in processed_files:
                svg_path = os.path.join(hand_images_folder, svg_file)
                metadata = extract_metadata(svg_path)
                selected_fingertip = tuple(map(int, metadata['Selected Fingertip'][1:-1].split(',')))  # Convert to tuple
                angle = float(metadata['Angle'].split()[0])  # Extract angle
                hand_color = metadata.get('Hand Color')
                #adjusted_angle = adjust_angle(angle)
                #adjusted_angle = normalize_angle(angle)
                # Randomly select an animation folder
                selected_animation_folder = random.choice(animation_folders)
                #selected_animation_frames = animation_folders[selected_animation_folder]
                print("selected animation folder: ", selected_animation_folder)
                print("Metadata:")
                print(metadata)
                display_animation_with_svg(svg_path, selected_animation_folder, angle, selected_fingertip, hand_color)
                processed_files.add(svg_file)  # Mark the file as processed
        time.sleep(1)  # Adjust the sleep time as needed

# Function to run the display function
def run_display():
    start_index = 0  # Initialize the starting index
    while True:
        subfolders = [os.path.join(SAVED_FRAMES_PATH, d) for d in os.listdir(SAVED_FRAMES_PATH) if os.path.isdir(os.path.join(SAVED_FRAMES_PATH, d))]

        if subfolders:
            valid_subfolders = []

            for subfolder in subfolders:
                files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
                # Check if the subfolder contains exactly 300 files
                if len(files) == 300:
                    valid_subfolders.append(subfolder)

            if valid_subfolders:
                # Ensure we have no more than 15 valid subfolders to display
                num_valid_subfolders = len(valid_subfolders)

                if num_valid_subfolders > 15:
                    # Update start index to ensure we show the last 15 subfolders
                    start_index = num_valid_subfolders - 15

                # Get the last 15 valid subfolders
                displayed_subfolders = valid_subfolders[start_index:start_index + 15]

                first_subfolder = displayed_subfolders[0]  # Get the first valid subfolder
                files = sorted([os.path.join(first_subfolder, f) for f in os.listdir(first_subfolder) if f.endswith('.png')])

                if files:
                    first_file = files[0]  # Get the first file in the first subfolder
                    first_file_load = Image.open(first_file)
                    file_width, file_height = first_file_load.size

                    new_width = file_width//4
                    new_height = file_height//4

                threads = []
                random.shuffle(grid_coordinates)
                for i, subfolder in enumerate(displayed_subfolders):
                    # Generate random coordinates for each subfolder
                    #x_coord = random.randint(0 + new_width, width_ - new_width)  # Assuming an approximate frame width
                    #y_coord = random.randint(0 + new_height, height_ - new_height)  # Assuming an approximate frame height

                    #print("grid_coordinates i: ", grid_coordinates[i])
                    # Use the shuffled list of coordinates sequentially for each subfolder
                    x_coord, y_coord = grid_coordinates[i]

                    # Create a new thread for each subfolder to display its animation frames independently
                    thread = threading.Thread(target=display_saved_animation, args=(subfolder, canvas, x_coord, y_coord, new_width, new_height, flag_fade_out))
                    thread.start()
                    threads.append(thread)

                # Wait for all threads to finish
                for thread in threads:
                    thread.join()

        time.sleep(1)  # Adjust the sleep time as needed to control the update frequency


def get_file_list(folder_id):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    return file_list


def get_most_recent_file(folder_id):
    file_list = get_file_list(folder_id)

    if not file_list:
        print("The folder is empty.")
        return None

    file_list.sort(key=lambda x: x['modifiedDate'], reverse=True)
    most_recent_file = file_list[0]

    print(f"Most recent file: {most_recent_file['title']} (Modified Date: {most_recent_file['modifiedDate']})")
    return most_recent_file


def download_file(file, local_folder):
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    file_path = os.path.join(local_folder, file['title'])
    file.GetContentFile(file_path)
    print(f"File downloaded to {file_path}")
    return file_path

def get_drive_files():
    previous_file_count = len(get_file_list(folder_id))

    while True:
        time.sleep(interval)
        current_file_list = get_file_list(folder_id)
        current_file_count = len(current_file_list)

        if current_file_count > previous_file_count:
            print(f"New file detected! Total files: {current_file_count}")
            most_recent_file = get_most_recent_file(folder_id)
            if most_recent_file:
                download_file(most_recent_file, local_folder)

        previous_file_count = current_file_count




if __name__ == '__main__':
    folder_id = '1IyhWRWJZ9KPyfJPC9f9xatIs-ac_cyLa'
    local_folder = 'hand_images_from_drive'
    interval = 1  # Check every 1 second
    #all_animation_frames = load_all_animation_frames()

    grid_coordinates = calculate_grid_coordinates(width_, height_, num_rows, num_columns)
    #print("cords: ", grid_coordinates)
    # Shuffle the list of grid coordinates
    #random.shuffle(grid_coordinates)
    #print("grid_coordinates 0: ", grid_coordinates[0])

    # Start the threads
    fetch_svg_thread = threading.Thread(target=fetch_svg_files)
    fetch_svg_thread.start()

    run_display_thread = threading.Thread(target=run_display)
    run_display_thread.start()

    get_drive_files_thread = threading.Thread(target=get_drive_files)
    get_drive_files_thread.start()

    root.mainloop()