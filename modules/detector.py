from pathlib import Path
from ultralytics import YOLO
from modules.waste_guide import normalize_category
class WasteDetector:
    def __init__(self,path,confidence):
        self.model=None; self.confidence=confidence
        if Path(path).exists():
            try: self.model=YOLO(str(path))
            except Exception: self.model=None
    @property
    def available(self): return self.model is not None
    def predict(self,image):
        if not self.available: raise RuntimeError("Add a trained model at models/best.pt")
        results=self.model.predict(source=image,conf=self.confidence,verbose=False)
        found=[]
        for result in results:
            for box in result.boxes or []:
                cls=int(box.cls.item()); raw=str(result.names[cls])
                cat=normalize_category(raw) or "general"
                found.append({"category":cat,"item_name":raw.replace("_"," ").title(),"confidence":float(box.conf.item()),"bbox":[float(v) for v in box.xyxy[0].tolist()]})
        return found
