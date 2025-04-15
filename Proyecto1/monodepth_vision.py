import sys
import gi
from gst_utils import gst_launch

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

# general
DEVICE="AUTO"

# paths reemplazar en maquina virtual
MODELS_PATH="/home/carolina/Proyecto1/model" 
MODELS_PROC_PATH="/home/carolina/Proyecto1/model"

# Models
MODEL_1="MiDaS_small"

# Model proc
HPE_MODEL_PROC=f"{MODELS_PROC_PATH}/{MODEL_1}.json"

# Model paths
HPE_MODEL=f"{MODELS_PATH}/{MODEL_1}.xml"

# Input
#INPUT="https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"
INPUT="/home/carolina/Proyecto1/Coco-Walking-in-Berkeley.mp4"

pipeline_str = (
    f'urisourcebin buffer-size=4096 uri={INPUT} ! '
    f'decodebin ! '
    f'gvaclassify model={HPE_MODEL} model-proc={HPE_MODEL_PROC} device={DEVICE} inference-region=full-frame ! queue ! '
    f'queue ! '
    f'gvawatermark ! videoconvert ! video/x-raw,format=BGR ! appsink'
)
#gst_launch(pipeline_str)
pipeline = Gst.parse_launch(pipeline_str)
pipeline.set_state(Gst.State.PLAYING)
