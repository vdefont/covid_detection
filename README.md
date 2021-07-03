TRAIN STEPS:

1. %run preprocess.py

- %time make_images("png", 224)
  - Takes 4 minutes
- preprocess_csv()
- make_id_orig_size_csv()
- make_metadata_feats(sname="train")

2. %run make_data_class.py

- create_and_validate_data(src="png224", extn="png", dst="png224", valid_amt=0.3)

3. %run make_data_detect.py

- make_boxes_png_224()

4. Train class

- Open vastai

- Upload:
  - data_class/png224.zip
  - const.py  
  - train_class.py
  - train_class.ipynb
  
- Run train_class.ipynb

- Download:
  - class/ models + preds
  - neg/ models + preds
  
5. Train detect

- Upload:
  - data_detect/png224.zip
  - const.py
  - train_detect.py
  - train_detect.ipynb
  
- Run train_detect.ipynb

- Download:
  - models (preds optional)
  
TEST STEPS

1 . %run preprocess.py

- %time make_images("png", 224, test_only=True)
  - takes 1 minute
- make_id_orig_size_csv(test_only=True)
- make_metadata_feats(sname="test")

2. %run make_data_class.py

- create_and_validate_data(src="png224", extn="png", dst="png224_test", valid_amt=0.3, test_only=True)

3. %run make_data_detect.py

- make_boxes_png_224(test_only=True)

4. Kaggle predictions

- Upload: All py + ipynb files