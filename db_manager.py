import json
import iris
from pathlib import Path

db_name = "db.json"

class Db:
    def __init__(self):
        self.path = Path(db_name)
        if not self.path.exists() or self.path.read_text().strip() == "":
            self.path.write_text("{}")
    
    # Stores the eye data in our db under a given name,
    # If data under that name already exists, we keep only the sharpest data
    def replace(self, name, code : iris.IrisTemplate , sharpness):
        serialized_template = code.serialize()

        newEye = {
            "code" : serialized_template,
            "sharpness" : sharpness
        }

        with self.path.open("r") as f:
            data = json.load(f)

        if name in data:
            oldEye = data[name]
            if newEye["sharpness"] < oldEye["sharpness"]:
                return
        
        data[name] = newEye

        with self.path.open("w") as f:
            json.dump(data, f, indent=2)


            
        

