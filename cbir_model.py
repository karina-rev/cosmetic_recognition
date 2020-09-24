import torch
import utils
import numpy as np
import faiss
import config
from torchvision import models, transforms, datasets

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    Source: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class CBIR():
    def __init__(self):
        self.transforms_ = transforms.Compose([
            transforms.Resize(size=[224, 224], interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.dataset = ImageFolderWithPaths(config.PATH_TO_FILES,
                                            self.transforms_,
                                            is_valid_file=utils.is_image_file,
                                            loader=utils.heic_loader)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1)

        self.model = models.resnet101(pretrained=True)
        self.image_paths = []
        self.descriptors = []

    def pooling_output(self, x):
        for layer_name, layer in self.model._modules.items():
            x = layer(x)
            if layer_name == 'avgpool':
                break
        return x

    def train(self):
        # извлекаем дескрипторы
        self.model.to(DEVICE)
        with torch.no_grad():
            self.model.eval()
            for inputs, labels, paths in self.dataloader:
                result = self.pooling_output(inputs.to(DEVICE))
                self.descriptors.append(result.cpu().view(1, -1).numpy())
                self.image_paths.append(paths)
                torch.cuda.empty_cache()

        # создаем Flat индекс и добавляем векторы дескрипторов
        index = faiss.IndexFlatL2(2048)
        descriptors = np.vstack(self.descriptors)
        index.add(descriptors)
        faiss.write_index(index, config.PATH_TO_FAISS_INDEX)
        return self

    def search_img(self, img):
        """
        Поиск похожих изображений
        :param img: изображение для поиска похожих
        :return: (distance, indices) для 9 элементов; остортированы по уменьшению расстрояния
        distance - расстояние от вектора данного изображения до похожих
        indices - индексы похожих изображений
        """
        index = faiss.read_index(config.PATH_TO_FAISS_INDEX)
        input_tensor = self.transforms_(img)
        input_tensor = input_tensor.view(1, *input_tensor.shape)
        with torch.no_grad():
            query_descriptors = self.pooling_output(input_tensor.to(DEVICE)).cpu().numpy()
            distance, indices = index.search(query_descriptors.reshape(1, 2048), 9)
        return distance, indices


