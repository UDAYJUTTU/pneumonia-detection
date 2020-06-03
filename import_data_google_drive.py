import zipfile

zip_file = zipfile.ZipFile('/content/stage_2_train_images.zip', 'r')
zip_file.extractall('/content/stage_2_train_images')
zip_file.close()

zip_file = zipfile.ZipFile('/content/stage_2_train_labels.csv.zip', 'r')
zip_file.extractall('/content/stage_2_train_labels.csv')
zip_file.close()

zip_file = zipfile.ZipFile('/content/stage_2_detailed_class_info.csv.zip', 'r')
zip_file.extractall('/content')
zip_file.close()

