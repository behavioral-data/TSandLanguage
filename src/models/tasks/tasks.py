from lightning import LightningDataModule
from src.utils import get_logger

logger = get_logger(__name__)

SUPPORTED_TASK_TYPES=[
    "classification",
    "autoencoder",
    "multimodal"
]

class Task(LightningDataModule):
    def __init__(self):
        super().__init__()
        for task_type in SUPPORTED_TASK_TYPES:
            setattr(self,f"is_{task_type}",False)

        # only computes full dataset if dataset getter methods are invoked
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_name(self):
        return self.__class__.__name__
    
    def get_description(self):
        raise NotImplementedError
    

    def get_description(self):
        return self.__doc__

    def get_train_dataset(self):
        raise NotImplementedError

    def get_val_dataset(self):
        raise NotImplementedError

    def get_test_dataset(self):
        raise NotImplementedError

    def get_labler(self):
        return NotImplementedError

    def get_labler(self):
        return NotImplementedError
    
    def cache(self):
        """If called, the task will look for a self.cache_path attribute and cache the dataset there"""
        raise NotImplementedError

    def get_metadata_lablers(self):
        return {}

    def get_metadata_types(self):
        return []

class TaskTypeMixin():
    def __init__(self):
        self.is_regression=False
        self.is_classification=False
        self.is_autoencoder=False
        self.is_double_encoding = False

class ClassificationMixin(TaskTypeMixin):
    def __init__(self):
        TaskTypeMixin.__init__(self)
        self.is_classification = True


class RegressionMixin(TaskTypeMixin):
    def __init__(self):
        TaskTypeMixin.__init__(self)
        self.is_regression=True

class AutoencodeMixin(RegressionMixin):
    def __init__(self):
        self.is_autoencoder = True
        super(AutoencodeMixin,self).__init__()

class MultimodalMixin(TaskTypeMixin):
    def __init__(self):
        self.is_multimodal = True
        super(MultimodalMixin,self).__init__()

