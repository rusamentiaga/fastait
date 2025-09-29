import fastait.io

def test_load_experiment_data():
    data = fastait.io.load_csv_folder("data/CFRP-006_facq-120Hz_s-Back_Img-2000", use_cache=True, verbose=True)
    assert data.shape == (2000, 512, 512)
    