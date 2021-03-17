import torch.utils.data
from datasets.aligned_dataset import AlignedDataset
from datasets.unaligned_dataset import UnalignedDataset
from datasets.single_dataset import SingleDataset

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        #from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset(opt)
    elif opt.dataset_mode == 'unaligned':
        #from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        #from data.single_dataset import SingleDataset
        dataset = SingleDataset()
        dataset.initialize(opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    return dataset


class CustomDatasetDataLoader():
    def __init__(self, opt):
        #super(CustomDatasetDataLoader,self).initialize(opt)
        #print("Opt.nThreads = ", opt.nThreads)
        self.opt        = opt
        kwargs          = {'num_workers': 1, 'pin_memory': True} if len(opt.gpu_ids) > 0 else {}
        self.dataset    = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(  self.dataset,
                                                        batch_size=opt.batchSize,
                                                        shuffle=not opt.serial_batches,
                                                        #num_workers=int(opt.nThreads),
                                                        **kwargs )
    def name(self):
        return 'CustomDatasetDataLoader'

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
