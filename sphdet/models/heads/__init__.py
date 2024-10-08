from .sph_rcnn_head import SphShared2FCBBoxHead, SphStandardRoIHead
from .sph_retina_head import SphRetinaHead
from .kent_retina_head import KentRetinaHead
from .sph_rpn_head import SphRPNHead
from .sph_ssd_head import SphSSDHead
from .sph_fcos_head import SphFCOSHead

__all__ = ['SphShared2FCBBoxHead', 'SphStandardRoIHead',
           'SphRetinaHead',
           'KentRetinaHead',
           'SphRPNHead',
           'SphSSDHead',
           'SphFCOSHead']
