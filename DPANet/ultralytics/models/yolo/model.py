from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel


class YOLO(Model):
    """YOLO detection model entry point for the DPANet-pruned project."""

    def __init__(self, model="ultralytics/cfg/models/DPANet/DPANet.yaml", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }
