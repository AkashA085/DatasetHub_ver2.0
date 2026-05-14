import pytest
from app.core.database import Dataset, Project, User, DatasetValidation, ClassDistribution, Image as DBImage
import uuid


def test_list_datasets(client, sample_dataset):
    response = client.get("/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert len(data["datasets"]) >= 1
    assert data["datasets"][0]["id"] == sample_dataset.id

def test_get_dataset_details(client, sample_dataset):
    response = client.get(f"/datasets/{sample_dataset.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == sample_dataset.id
    assert data["total_images"] == 10

def test_get_nonexistent_dataset(client):
    response = client.get(f"/datasets/{uuid.uuid4()}")
    assert response.status_code == 404

def test_delete_dataset(client, sample_dataset):
    # Store ID before deletion to avoid ObjectDeletedError
    dataset_id = sample_dataset.id
    response = client.delete(f"/datasets/{dataset_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verify it's gone from DB
    response = client.get(f"/datasets/{dataset_id}")
    assert response.status_code == 404

def test_list_datasets_with_params(client, sample_dataset, db_session):
    # Add another dataset
    ds2 = Dataset(id=str(uuid.uuid4()), project_id=sample_dataset.project_id, format_type="coco")
    db_session.add(ds2)
    db_session.commit()
    
    response = client.get("/datasets", params={"format_type": "coco", "sort_by": "created_at", "order": "asc"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["datasets"]) >= 1
    # Find ds2 in results
    ds2_res = next((d for d in data["datasets"] if d["id"] == ds2.id), None)
    assert ds2_res is not None
    assert ds2_res["format_type"] == "coco"

def test_get_dataset_details_full(client, sample_dataset, db_session):
    # Add validation info
    val = DatasetValidation(
        dataset_id=sample_dataset.id, total_images=10, total_labels=10, 
        missing_labels=0, orphan_labels=0, empty_labels=0, corrupted_images=0,
        class_ids_found=[0, 1]
    )
    db_session.add(val)
    # Add class distribution
    cd = ClassDistribution(dataset_id=sample_dataset.id, class_id="0", object_count=5)
    db_session.add(cd)
    db_session.commit()
    
    response = client.get(f"/datasets/{sample_dataset.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["validation"] is not None
    assert len(data["class_distribution"]) >= 1

def test_list_dataset_images(client, sample_dataset, db_session):
    # Add an image
    img = DBImage(id=str(uuid.uuid4()), dataset_id=sample_dataset.id, file_name="img1.jpg")
    img.has_label = True
    db_session.add(img)
    db_session.commit()
    
    response = client.get(f"/datasets/{sample_dataset.id}/images", params={"has_label": True})
    assert response.status_code == 200
    data = response.json()
    assert len(data["images"]) >= 1
    assert data["images"][0]["has_label"] == True

def test_delete_dataset_not_found(client):
    response = client.delete(f"/datasets/{uuid.uuid4()}")
    assert response.status_code == 404
