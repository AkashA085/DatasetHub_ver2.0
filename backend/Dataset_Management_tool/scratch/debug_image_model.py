from app.core.database import Image
import uuid

try:
    img = Image(id=str(uuid.uuid4()), dataset_id="test", file_name="test.jpg")
    print("Success")
except TypeError as e:
    print(f"Failed: {e}")
