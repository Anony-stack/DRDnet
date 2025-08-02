from .eva_original import eva02_large_patch14_clip_224
from .eval_moment import EVA_moment
__factory = {
    'eva02_l':eva02_large_patch14_clip_224, 
}


def build_model(config, num_classes):
    model_type = config.MODEL.TYPE
    if model_type == 'eval_moment':
        model = EVA_moment(config, num_classes)
    else:
        model = __factory[config.MODEL.NAME](pretrained=True, num_classes=num_classes)
    
    return model
