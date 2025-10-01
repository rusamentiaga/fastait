import os
import pytest
import fastait.io

def test_load_experiment_data():
    folder_path = "data/CFRP-006_facq-120Hz_s-Back_Img-2000"
    
    if not os.path.exists(folder_path):
        pytest.skip(f"Folder {folder_path} does not exist, skipping test.")
    
    data = fastait.io.load_csv_folder(folder_path, use_cache=True, verbose=True)
    assert data.shape == (2000, 512, 512)
    
    