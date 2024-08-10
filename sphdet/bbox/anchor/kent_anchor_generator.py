import torch
from mmdet.core.anchor import AnchorGenerator
from mmdet.core.anchor.builder import ANCHOR_GENERATORS

from sphdet.bbox.box_formator import Planar2SphBoxTransform, SphBox2KentTransform, Planar2KentTransform
import pdb

from line_profiler import LineProfiler
import threading
import time

def profile_function(func, interval=10):
    profiler = LineProfiler()
    profiler.add_function(func)
    profiler.enable_by_count()

    while True:
        time.sleep(interval)
        profiler.print_stats()
        profiler.disable()
        profiler = LineProfiler()
        profiler.add_function(func)
        profiler.enable_by_count()

@ANCHOR_GENERATORS.register_module()
class KentAnchorGenerator(AnchorGenerator):
    """Spherical anchor generator for 2D anchor-based detectors.

    Horizontal bounding box represented by (theta, phi, alpha, beta).
    """
    def __init__(self, box_formator='sph2pix', box_version=4, *args, **kwargs):
        super(KentAnchorGenerator, self).__init__(*args, **kwargs)
        assert box_formator in ['sph2pix', 'pix2sph', 'sph2tan', 'tan2sph']
        assert box_version in [4, 5]
                
        #self.box_formator = Planar2SphBoxTransform(box_formator, box_version)
        #profiling_thread = threading.Thread(target=profile_function, args=(Planar2KentTransform.__call__, 100))
        #profiling_thread.start()
        self.box_formator = Planar2KentTransform(box_formator, box_version)
        #sif (box_version==5):
        #self.box_formator_kent = SphBox2KentTransform()

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda'):
        anchors = super(KentAnchorGenerator, self).single_level_grid_priors(featmap_size, level_idx, dtype, device)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        img_h, img_w = feat_h * stride_h, feat_w * stride_w

        sph_anchors = self.box_formator(anchors, (img_h, img_w))
        #pdb.set_trace()
        #kent_anchors = self.box_formator_kent(sph_anchors, (img_h, img_w))
        #pdb.set_trace()
        return sph_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        anchors = super(KentAnchorGenerator, self).single_level_grid_anchors(base_anchors, featmap_size, stride, device)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = stride
        img_h, img_w = feat_h * stride_h, feat_w * stride_w

        sph_anchors = self.box_formator(anchors, (img_h, img_w))
        #kent_anchors = self.box_formator_kent(sph_anchors, (img_h, img_w))
        return sph_anchors
       