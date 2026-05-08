from app.core.database import SessionLocal, Dataset

db = SessionLocal()
print('dataset count', db.query(Dataset).count())
for ds in db.query(Dataset).all():
    print(ds.id, ds.name, ds.total_images)
db.close()