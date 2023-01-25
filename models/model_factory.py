from models import ResNet
from models import classifier
from models import ResNet_SN
from models import ResNet_SRN

encoders_map = {
    'resnet18': ResNet.resnet18,
    'resnet50': ResNet.resnet50,
    'convnet': ResNet.ConvNet,
    'resnet18_sn':ResNet_SN.resnet18,
    'resnet18_srn':ResNet_SRN.resnet18
}

classifiers_map = {
    'base': classifier.Classifier,
}

projectionhead_map = {
    'mlp':classifier.ProjectionHead,
}

def get_encoder(name):
    if name not in encoders_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return encoders_map[name](**kwargs)

    return get_network_fn


def get_encoder_from_config(config):
    return get_encoder(config["name"])()


def get_classifier(name):
    if name not in classifiers_map:
        raise ValueError('Name of classifier unknown %s' % name)

    def get_network_fn(**kwargs):
        return classifiers_map[name](**kwargs)

    return get_network_fn

def get_classifier_from_config(config):
    return get_classifier(config["name"])(
        in_dim=config["in_dim"],
        num_classes=config["num_classes"]
    )

### Added
def get_projectionhead(name):
    if name not in projectionhead_map:
        raise ValueError('Name of projection head unknown %s' % name)

    def get_network_fn(**kwargs):
        return projectionhead_map[name](**kwargs)

    return get_network_fn

def get_projectionhead_from_config(config):
    return get_projectionhead(config["name"])(
        embedding_dim=config["embedding_dim"],
        projection_dim=config["projection_dim"],
        dropout=config["dropout"] 
    )