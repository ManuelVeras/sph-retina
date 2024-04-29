_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_120e.py',
    '../_base_/datasets/indoor360.py',
    '../_base_/models/sph_retinanet_r50_fpn.py',
]

# log
checkpoint_config = dict(interval=25)
evaluation = dict(interval=5)

#optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=5, norm_type=2))

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')])

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2)


model = dict(
    train_cfg=dict(
        assigner=dict(
            iou_calculator=dict(
                backend='sph2pob_standard_iou')),),
    test_cfg=dict(
        nms=dict(iou_threshold=0.5),
        iou_calculator='naive_iou',
        box_formator='sph2pix'))