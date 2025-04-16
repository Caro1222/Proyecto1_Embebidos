import sys

from gst_utils import gst_launch

# general
DEVICE="AUTO"

# paths
MODELS_PATH="/home/carolina/Proyecto1_Embebidos"
MODELS_PROC_PATH="/home/carolina/Proyecto1_Embebidos"

# Models
MODEL_1="human-pose-estimation-0001"

# Model proc
HPE_MODEL_PROC=f"{MODELS_PROC_PATH}/{MODEL_1}.json"

# Model paths
HPE_MODEL=f"{MODELS_PATH}/intel/{MODEL_1}/FP32/{MODEL_1}.xml"

# Input
INPUT="https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"

pipeline_str = (
    f'urisourcebin buffer-size=4096 uri={INPUT} ! '
    f'decodebin ! '
    f'gvaclassify model={HPE_MODEL} model-proc={HPE_MODEL_PROC} device={DEVICE} inference-region=full-frame ! queue ! '
    f'queue ! '
    f'gvawatermark ! videoconvert ! video/x-raw,format=BGR ! appsink'
)

gst_launch(pipeline_str)
