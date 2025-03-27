import numpy as np
import argparse
import cv2
import os
from tkinter import Tk, filedialog, Button, Label, Frame
from PIL import Image, ImageTk

# Paths to load the model
DIR = r"C:\Users\HP\OneDrive\Desktop\color"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Define a function to process the image
def process_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not filepath:
        return

    image = cv2.imread(filepath)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    print("Colorizing the image")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    # Combine the original and colorized images side by side for display
    combined_image = np.hstack((image, colorized))
    combined_image = cv2.resize(combined_image, (700, 350))

    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(combined_image)
    imgtk = ImageTk.PhotoImage(img)

    image_label.config(image=imgtk)
    image_label.image = imgtk
    status_label.config(text="Image Processed Successfully!")

# Create the GUI
root = Tk()
root.title("Image Colorization Tool")
root.geometry("800x600")
root.configure(bg="#f4f4f4")

# Create a frame for the title
title_frame = Frame(root, bg="#333333", height=80)
title_frame.pack(fill="x")
title_label = Label(
    title_frame,
    text="Image Colorization Tool",
    fg="white",
    bg="#333333",
    font=("Arial", 20, "bold")
)
title_label.pack(pady=20)

# Create a frame for the buttons
button_frame = Frame(root, bg="#f4f4f4")
button_frame.pack(pady=20)
select_button = Button(
    button_frame,
    text="Select Black-and-White Image",
    command=process_image,
    bg="#008CBA",
    fg="white",
    font=("Arial", 12, "bold"),
    padx=10,
    pady=5
)
select_button.pack()

# Add a label to display the image
image_label = Label(root, bg="#f4f4f4")
image_label.pack(pady=20)

# Add a status label
status_label = Label(
    root,
    text="Welcome to the Image Colorization Tool!",
    bg="#f4f4f4",
    fg="#333333",
    font=("Arial", 12)
)
status_label.pack(pady=10)

# Start the GUI event loop
root.mainloop()
