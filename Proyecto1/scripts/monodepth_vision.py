import os
from gst_utils import gst_launch

# general
DEVICE="AUTO"

# paths reemplazar en maquina virtual
MODELS_PATH="/home/carolina/Proyecto1_Embebidos/Proyecto1/model" 
MODELS_PROC_PATH="/home/carolina/Proyecto1_Embebidos/Proyecto1/model"

# Models
MODEL_1="MiDaS_small"

# Model proc
HPE_MODEL_PROC=f"{MODELS_PROC_PATH}/{MODEL_1}.json"

# Model paths
HPE_MODEL=f"{MODELS_PATH}/{MODEL_1}.xml"

# Input
#INPUT="https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"
INPUT="/home/carolina/Proyecto1_Embebidos/Proyecto1/inputs/Coco-Walking-in-Berkeley.mp4"

OUTPUT_PATH = "/home/carolina/Proyecto1_Embebidos/Proyecto1/output/salida.mp4"

pipeline_str = (
    f'urisourcebin buffer-size=4096 uri={INPUT} ! '
    f'decodebin ! '
    f'gvaclassify model={HPE_MODEL} model-proc={HPE_MODEL_PROC} device={DEVICE} inference-region=full-frame ! queue ! '
    f'queue ! '
    f'gvawatermark ! videoconvert ! video/x-raw,format=BGR ! appsink'
)

# Ejecutar pipeline
gst_launch(pipeline_str)
