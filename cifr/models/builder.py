from ..core.registry import Registry

DATASETS = Registry('datasets')
ARCHITECTURES = Registry('architectures')
DISCRIMINATORS = Registry('discriminators')
LOSSES = Registry('losses')
OPTIMIZERS = Registry('optimizers')
MODELS = Registry('models')


def build_dataset(cfg):
    return DATASETS.build(cfg)

def build_architecture(cfg):
    return ARCHITECTURES.build(cfg)

def build_discriminator(cfg):
    return DISCRIMINATORS.build(cfg)

def build_loss(cfg):
    return LOSSES.build(cfg)

def build_optimizer(cfg):
    return OPTIMIZERS.build(cfg)

def build_model(cfg):
    return MODELS.build(cfg)
