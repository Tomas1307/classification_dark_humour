import os
import tkinter as tk
from tkinter import Label, Button, Scale
from PIL import Image, ImageTk
import pandas as pd
import re

# Ruta de las imágenes
image_dir = "data/memes_conjuntos"
csv_path = "data/textos/chistes_clasificados/clasificacion_memes.csv"

# Cargar progreso
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    reviewed_ids = set(df['id'].astype(str).tolist())  # Convertimos a string para coincidir con el formato del ID de imagen
else:
    df = pd.DataFrame(columns=["id", "descripcion", "label", "nivel_risa"])
    reviewed_ids = set()

# Obtener y ordenar lista de imágenes numéricamente
def extract_number(filename):
    match = re.search(r'\d+', filename)  # Extrae solo el número de la imagen
    return int(match.group()) if match else float('inf')

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files = sorted(image_files, key=extract_number)  # Ordena numéricamente

# Filtra las imágenes que no han sido revisadas
image_files = [f for f in image_files if os.path.splitext(f)[0] not in reviewed_ids]

# Configurar ventana
root = tk.Tk()
root.title("Clasificación de Memes")

# Variables para controlar la interfaz
image_label = Label(root)
image_label.pack()
status_label = Label(root, text="")
status_label.pack()

# Control deslizante para el nivel de risa
laugh_scale = Scale(root, from_=1, to=5, orient="horizontal", label="Nivel de risa")
laugh_scale.pack()

# Función para guardar y mostrar siguiente imagen
def save_and_next(label, nivel_risa):
    """
    Saves the current image classification to a CSV file and displays the next image in the list.
    If all images have been reviewed, it disables the buttons and shows a completion message.

    Args:
        label (bool): Indicates if the image is funny (True) or not (False).
        nivel_risa (int): The laughter level rating assigned to the image (1-5).
    """
    global current_image_index
    # Guardar en CSV
    image_file = image_files[current_image_index]
    image_id = os.path.splitext(image_file)[0]  # Obtener el ID sin la extensión
    df.loc[len(df)] = [image_id, image_file, label, nivel_risa]
    df.to_csv(csv_path, index=False)
    
    # Mostrar siguiente imagen
    current_image_index += 1
    if current_image_index < len(image_files):
        show_image()
    else:
        status_label.config(text="¡Clasificación completa!")
        yes_button.config(state="disabled")
        no_button.config(state="disabled")

# Función para mostrar la imagen actual
def show_image():
    """
    Displays the current joke in the list with its ID and the review progress status.
    """
    image_file = image_files[current_image_index]
    img_path = os.path.join(image_dir, image_file)
    img = Image.open(img_path)
    img.thumbnail((800, 800))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img
    status_label.config(text=f"Mostrando: {image_file} ({current_image_index + 1}/{len(image_files)})")

# Función de evento para calificación con teclas del 1 al 5
def on_key_press(event):
    """
    Event handler for key presses to allow users to rate the joke using keys 1 to 5.
    
    Args:
        event: The key press event capturing the user's input.
    """
    if event.char in "12345":
        nivel_risa = int(event.char)
        save_and_next(label=True, nivel_risa=nivel_risa)

# Botones de clasificación
yes_button = Button(root, text="Gracioso", command=lambda: save_and_next(True, laugh_scale.get()))
yes_button.pack(side="left")
no_button = Button(root, text="No gracioso", command=lambda: save_and_next(False, 0))
no_button.pack(side="right")

# Asignar la función on_key_press a los eventos de teclado
root.bind("<Key>", on_key_press)

# Iniciar
current_image_index = 0
show_image()

root.mainloop()
