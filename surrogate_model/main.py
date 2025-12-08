from surrogate_model.model import SurrogateModel
from surrogate_model.data_generation import SurrogateDatasetBuilder
from surrogate_model.train import train_surrogate

if __name__ == "__main__":
    surrogate_model = SurrogateModel()
    surrogate_dataset_builder = SurrogateDatasetBuilder()
    train_surrogate(surrogate_model, surrogate_dataset_builder)