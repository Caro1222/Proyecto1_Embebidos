import os
import time
from pathlib import Path
import requests

import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import openvino.properties as props

from notebook_utils import (
    download_file,
    load_image,
    collect_telemetry,
    device_widget,
)

# Telemetría (opcional, puedes comentarlo si no lo deseas)
collect_telemetry("vision-monodepth.py")

# Configurar modelo
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)

ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder)
download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder)

model_xml_path = model_folder / ir_model_name_xml

# Funciones auxiliares
def normalize_minmax(data):
    return (data - data.min()) / (data.max() - data.min())

def convert_result_to_image(result, colormap="viridis"):
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    return result.astype(np.uint8)

def to_rgb(image_data):
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

# Inicializar OpenVINO
device = device_widget().value
cache_folder = Path("cache")
cache_folder.mkdir(exist_ok=True)

core = ov.Core()
core.set_property({props.cache_dir(): cache_folder})
model = core.read_model(model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
network_image_height, network_image_width = input_key.shape[2:]

# Imagen de prueba
IMAGE_URL = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg"
IMAGE_NAME = "coco_bike.jpg"
image = load_image(IMAGE_NAME, IMAGE_URL)

resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))
input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
result = compiled_model([input_image])[output_key]
result_image = convert_result_to_image(result)
result_image = cv2.resize(result_image, image.shape[:2][::-1])

# Mostrar resultados (solo si corres el script de forma interactiva)
plt.figure(figsize=(20, 15))
plt.subplot(1, 2, 1)
plt.imshow(to_rgb(image))
plt.subplot(1, 2, 2)
plt.imshow(result_image)
plt.show()

# Configuración de video
VIDEO_FILE = "Coco-Walking-in-Berkeley.mp4"
download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4",
    VIDEO_FILE,
)

NUM_SECONDS = 4
ADVANCE_FRAMES = 2
SCALE_OUTPUT = 0.5
FOURCC = cv2.VideoWriter_fourcc(*"vp09")

cap = cv2.VideoCapture(str(VIDEO_FILE))
ret, image = cap.read()
if not ret:
    raise ValueError(f"No se puede leer el video en {VIDEO_FILE}")
input_fps = cap.get(cv2.CAP_PROP_FPS)
input_video_frame_height, input_video_frame_width = image.shape[:2]

target_fps = input_fps / ADVANCE_FRAMES
target_frame_height = int(input_video_frame_height * SCALE_OUTPUT)
target_frame_width = int(input_video_frame_width * SCALE_OUTPUT)
cap.release()

print(f"Resolución original: {input_video_frame_width}x{input_video_frame_height}, {input_fps:.2f} FPS")
print(f"Resolución procesada: {target_frame_width}x{target_frame_height}, {target_fps:.2f} FPS")

# Salida de video
output_directory = Path("output")
output_directory.mkdir(exist_ok=True)
result_video_path = output_directory / f"{Path(VIDEO_FILE).stem}_monodepth.mp4"

cap = cv2.VideoCapture(str(VIDEO_FILE))
out_video = cv2.VideoWriter(
    str(result_video_path),
    FOURCC,
    target_fps,
    (target_frame_width * 2, target_frame_height),
)

num_frames = int(NUM_SECONDS * input_fps)
input_video_frame_nr = 0
start_time = time.perf_counter()
total_inference_duration = 0

try:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret or input_video_frame_nr >= num_frames:
            break

        resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

        inference_start_time = time.perf_counter()
        result = compiled_model([input_image])[output_key]
        inference_duration = time.perf_counter() - inference_start_time
        total_inference_duration += inference_duration

        result_frame = to_rgb(convert_result_to_image(result))
        result_frame = cv2.resize(result_frame, (target_frame_width, target_frame_height))
        image = cv2.resize(image, (target_frame_width, target_frame_height))
        stacked_frame = np.hstack((image, result_frame))
        out_video.write(stacked_frame)

        input_video_frame_nr += ADVANCE_FRAMES
        cap.set(cv2.CAP_PROP_POS_FRAMES, input_video_frame_nr)

except KeyboardInterrupt:
    print("Interrupción del usuario.")
finally:
    cap.release()
    out_video.release()
    duration = time.perf_counter() - start_time
    processed_frames = num_frames // ADVANCE_FRAMES
    print(
        f"Procesados {processed_frames} cuadros en {duration:.2f} segundos. "
        f"FPS total: {processed_frames/duration:.2f}. "
        f"Inference FPS: {processed_frames/total_inference_duration:.2f}"
    )
    print(f"Video guardado en: {result_video_path.resolve()}")


