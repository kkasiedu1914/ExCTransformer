from pytorch_lightning import LightningDataModule
from exctransformer.datamodule.batch_data import GraphDataset, create_data_loaders

class DataModule(LightningDataModule):
    def __init__(self, config):

        super().__init__()
        
        self.dataset_path = config['dataset_path']
        self.batch_size = config['batch_size']
        self.train_ratio = config['train_ratio']
        self.val_ratio = config['val_ratio']
        self.test_ratio = config['test_ratio']
        self.train_size = config['train_size']
        self.val_size = config['val_size']
        self.test_size = config['test_size']
        self.sub_data = config['sub_data']
        self.zipped = config['zipped']
        self.lazy = config['lazy']
        self.transform = config['transform']
        self.num_workers = config['num_workers']
        self.predict_set = config['predict_set']
        self.include_additional_features = not config.get("graph_only_mode", False)
      
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.predict_loader = None
        
        
    def setup(self, stage=None):
        """
        Set up the dataset and create data loaders for training, validation, and testing.

        Args:
        if self.train_size is not None and self.val_size is not None and self.test_size is not None:
        """
        self.dataset = GraphDataset(self.dataset_path, self.zipped, self.lazy, self.transform)
        if not self.predict_set:
            if (self.train_size is not None) or (self.val_size is not None) or (self.test_size is not None):

                self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                    self.dataset, 
                    train_size=self.train_size, 
                    val_size=self.val_size, 
                    test_size=self.test_size, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    num_workers=self.num_workers,
                    sub_data=self.sub_data,
                    include_additional_features=self.include_additional_features
                )
            else:
                self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                self.dataset, 
                train_ratio=self.train_ratio, 
                val_ratio=self.val_ratio, 
                test_ratio=self.test_ratio, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers,
                sub_data=self.sub_data,
                include_additional_features=self.include_additional_features
            )
        else:
            self.predict_loader =  create_data_loaders(
                    self.dataset, 
                    train_size=self.train_size, 
                    val_size=self.val_size, 
                    test_size=self.test_size, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    num_workers=self.num_workers,
                    predict_set=self.predict_set,
                    sub_data=self.sub_data,
                    include_additional_features=self.include_additional_features
                )     

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
    
    def predict_dataloader(self):
        return self.predict_loader
