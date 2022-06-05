import os
import time
import argparse
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger 

def set_config(inference, network_type=None):
    tim = time.strftime("_%m_%d_%H_%M", time.localtime())
    cfg = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cfg.model_save_path = "./VisionAlgo/trained_models/"

    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.gpu_id = [0]

    # robot
    cfg.robot_type = "panda"

    # dataset
    cfg.data_path = "/home/lihua/Desktop/Datasets/DREAM/synthetic/panda_synth_test_dr"
    # cfg.data_path = "/workspace/data/panda_synth_train_dr"
    cfg.training_data_proportion = 0.8

    if network_type == None:
        cfg.network_type = 'keypoint_rcnn_finetune'
    else:
        cfg.network_type = network_type

    
    if cfg.network_type == 'resnet':
        # model
        cfg.backbone_type = 'resnet50'
        cfg.backbone_n_keypoints = 7
        cfg.backbone_pretrained = True
        cfg.save_model_name = os.path.join(cfg.model_save_path, cfg.network_type + tim + '.pkl')

        # train
        cfg.epochs = 20
        cfg.batch_size = 3
        cfg.lr = 1e-4
        cfg.lr_scheduler_step_size = 5
        cfg.lr_scheduler_gamma = 0.5
        cfg.num_worker = 8

    elif cfg.network_type == 'keypoint_rcnn':
        # model
        cfg.backbone_type = 'keypointrcnn_resnet50_fpn'
        cfg.backbone_n_keypoints = 7
        cfg.backbone_pretrained = True
        cfg.save_model_name = os.path.join(cfg.model_save_path, cfg.network_type + tim + '.pkl')

        # train
        cfg.epochs = 12
        cfg.batch_size = 9
        cfg.lr = 1e-4
        cfg.lr_scheduler_step_size = 3
        cfg.lr_scheduler_gamma = 0.5
        cfg.num_worker = 0

    elif cfg.network_type == 'keypoint_rcnn_finetune':
        # model
        cfg.backbone_type = 'keypoint_rcnn_finetune'
        cfg.output_type = 'cross_entropy heatmap'  # 'gaussian heatmap' or 'cross_entropy heatmap'
        cfg.backbone_n_keypoints = 7
        cfg.backbone_pretrained = True
        cfg.save_model_name = os.path.join(cfg.model_save_path, cfg.network_type + tim + '.pkl')

        # train
        cfg.epochs = 20
        cfg.batch_size = 2
        cfg.lr = 1e-4
        cfg.lr_scheduler_step_size = 3
        cfg.lr_scheduler_gamma = 0.5
        cfg.num_worker = 0


    if not inference:
        # log
        cfg.print_interval = 100
        cfg.log_path = './VisionAlgo/logs/'
        cfg.log_name = 'training'+ tim +'.log'
        cfg.logger = get_logger(os.path.join(cfg.log_path, cfg.log_name))
        cfg.tensorboard_writer = SummaryWriter()

    return cfg


if __name__ == "__main__":
    pass