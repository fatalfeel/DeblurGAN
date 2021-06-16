from datasets.custom_dataset_data_loader import CustomDatasetDataLoader

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader(opt)
    print(data_loader.name())

    return data_loader