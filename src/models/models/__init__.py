# from src.models.models.models import *
# from src.models.models.bases import *
from src.models.models.vit import ViTForClassification

from torch import nn
from src.models.models.inceptiontime import InceptionTime
from src.models.models.transcription import HFTranscriptionModel
from src.models.models.multimodal_llm import LLaVA