import data.img_transforms as T
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset, VideoDataset
from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler
from data.datasets.deepchange import DeepChange
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC
from torch.utils.data import ConcatDataset, DataLoader


__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
}



def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))
    if config.DATA.DATASET == 'vcclothes':
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, mode=config.DATA.MODE)
    else:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT)

    return dataset


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.IMG_HEIGHT , config.DATA.IMG_WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.IMG_HEIGHT , config.DATA.IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test



def build_dataloader(config):
    dataset = build_dataset(config)
    # image dataset
    transform_train, transform_test = build_img_transforms(config)
    # transform_train = build_transform(config,is_train=True)
    # transform_test = build_transform(config,is_train=False)
    train_sampler = DistributedRandomIdentitySampler(dataset.train,
                                                     num_instances=config.DATA.NUM_INSTANCES,
                                                     seed=config.SOLVER.SEED)
    trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train, dataset_name=config.DATA.DATASET),
                             sampler=train_sampler,
                             batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=config.DATA.PIN_MEMORY, drop_last=True)

    galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test, dataset_name=config.DATA.DATASET),
                               sampler=DistributedInferenceSampler(dataset.gallery),
                               batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                               pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False)
    if config.DATA.DATASET == 'prcc':
        queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test, dataset_name=config.DATA.DATASET),
                                 sampler=DistributedInferenceSampler(dataset.query_same),
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=config.DATA.PIN_MEMORY,drop_last=False, shuffle=False)
        queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test, dataset_name=config.DATA.DATASET),
                                 sampler=DistributedInferenceSampler(dataset.query_diff),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)


        combined_dataset = ConcatDataset([queryloader_diff.dataset, galleryloader.dataset])

        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False
        )

        combined_dataset_same = ConcatDataset([queryloader_same.dataset, galleryloader.dataset])

        val_loader_same = DataLoader(
            dataset=combined_dataset_same,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False
        )

        return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same
    else:
        queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test, dataset_name=config.DATA.DATASET),
                                 sampler=DistributedInferenceSampler(dataset.query),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)

        combined_dataset = ConcatDataset([queryloader.dataset, galleryloader.dataset])

        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False
        )



        return trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader

    

    
