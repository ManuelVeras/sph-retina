_base_ = ['./sph_retinanet_r50_fpn.py',]

model = dict(
    bbox_head=dict(
        box_version=5,
        num_classes=47,
        anchor_generator=dict(
            box_version=5,),
        bbox_coder=dict(
            type='DeltaXYWHASphBBoxCoder',
            target_means=[.0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]),),
    train_cfg=dict(
        assigner=dict(
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0,
            iou_calculator=dict(backend = 'sph2pob_efficient_iou',
                box_version=5)),),
    test_cfg=dict(
        nms=dict(
            _delete_=True,
            type='nms_rotated',
            clockwise=False,)))   