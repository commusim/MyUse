conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        'dataset_path': "E:\python\data\gait\\64x64",
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
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 8),
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,

        # 'model_name': 'GaitSet_(8,8)',
        'model_name': 'GaitSet_Half_(8,8)',
        # 'model_name': 'GaitSet_Half_Fusion_(8,8)',
        # 'model_name': 'GaitSet_HPP_(8,8)',

        # 'model_name': 'GaitLocal_(8,8)',
        # 'model_name': 'GaitLocal_part_(8,8)',

        # 'model_name': 'GaitPart_(8,8)',
        # 'model_name': 'GaitPart_Half_(8,8)',
        # 'model_name': 'GaitLocal_HPP_(8,8)',

        # 'model_name': 'GaitSA_(8,8)',
        # 'model_name': 'GaitSA_prior_(8,8)',

    },
}
