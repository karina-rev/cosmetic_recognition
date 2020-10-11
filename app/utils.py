from torchvision import models, transforms
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"])
adapter = HTTPAdapter(max_retries=retry_strategy)


def pooling_output(x):
    model = models.resnet101(pretrained=True)
    for layer_name, layer in model._modules.items():
        x = layer(x)
        if layer_name == 'avgpool':
            break
    return x


def transform_torch():
    return transforms.Compose([
        transforms.Resize(size=[224, 224], interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
