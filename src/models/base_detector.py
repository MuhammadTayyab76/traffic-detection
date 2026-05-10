from abc import ABC,abstractmethod
import numpy as np

class BaseDetector(ABC):
    def __init__(self,weights_path):
        self.weights_path=weights_path
        self.model=None
        self.load_model()
        
    @abstractmethod
    def load_model(self):
        """Loads the model weights into memory."""
        pass
        
    @abstractmethod
    def detect(self,frame):
        """
        Takes a single cv2 frame.
        Must return a list of dictionaries containing:
        [{'bbox':[x1,y1,x2,y2], 'confidence':float, 'class_id':int, 'class_name':str}]
        """
        pass