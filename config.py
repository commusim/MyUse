conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        'dataset_path': "/home/ubuntu/commusim/dataset/GaitDatasetB-silh/Gaitset",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,  # train test区分
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 4e-5,
        'hard_or_full_trip': 'full',
        'batch_size': (4, 16),
        'restore_iter': 80000,
        'total_iter': 200000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        # 'model_name': 'GaitSet',
        'model_name': 'LegModel',
    },
}
