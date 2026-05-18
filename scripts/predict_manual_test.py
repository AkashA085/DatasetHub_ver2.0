import requests

# Test the predict endpoint
files = {
    'dataset_id': (None, '5a408f6d-28d5-425d-8ce7-22dd08e75035'),
    'image_file': open(r'd:\datasethub\datasethub_storage\uploads\5a408f6d-28d5-425d-8ce7-22dd08e75035\images\images\_A_fw_09_12AM_1.jpg', 'rb')
}

try:
    response = requests.post('http://localhost:8000/train/predict', files=files)
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        predictions = data.get('predictions', [])
        print(f'Predictions: {len(predictions)}')
        for pred in predictions:
            print(f'  Class: {pred["class"]}, Conf: {pred["confidence"]:.2%}')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Error: {e}')