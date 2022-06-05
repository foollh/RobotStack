import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import robotDataSets, collate_fn
from network import MyModel
from config import set_config
from torch.utils.data.distributed import DistributedSampler

def calculate_metrics(detections, targets):
    '''
    dst_keypoint: List(Tensor(N, 7, 3))
    gt_keypoints: List(Tensor(1, 7, 3))
    gt_roi: List(Tensor(1, 4))
    '''
    result = {}
    
    # object keypoint similarity
    similarity = []
    for detection, target in zip(detections, targets):

        dst_keypoints, dst_scores, gt_keypoints, gt_roi = detection['keypoints'], detection['scores'], target['keypoints'],  target['boxes']

        if dst_keypoints.shape[0] == 0:
            similarity.append(0)
        else:    
            score_index = torch.argmax(dst_scores)
            _, vis_index = torch.where(gt_keypoints[:, :, -1] == 1)

            # area
            src_area = (gt_roi[:, 2] - gt_roi[:, 0] + 1) * (gt_roi[:, 3] - gt_roi[:, 1] + 1)

            # measure the per-keypoint distance if keypoints visible
            dx = dst_keypoints[score_index, vis_index, 0] - gt_keypoints[0, vis_index, 0]
            dy = dst_keypoints[score_index, vis_index, 1] - gt_keypoints[0, vis_index, 1]

            e = (dx**2 + dy**2) / (src_area + np.spacing(1))
            e = torch.sum(torch.exp(-e), dim=0) / e.shape[0]
            similarity.append(e.item())
    
    result['oks'] = sum(similarity)/len(similarity)

    return result


def resnet50_train(cfg, model, criterion, optimizer, scheduler, train_loader, test_loader):
    min_loss = float('inf')
    for epoch in range(cfg.epochs):
        # train
        train_loss = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(cfg.device)
            targets = labels['keypoints_heatmaps'].to(cfg.device)

            # forward
            pred = model(images)
            loss = criterion(pred, targets)
            train_loss.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % cfg.print_interval == 0:
                cfg.logger.info("Epoch:[{0}/{1}] Training Loss: {2}".format(epoch+1, cfg.epochs, sum(train_loss)/len(train_loss)))

        cfg.logger.info("Epoch:[{0}/{1}] Training Loss: {2}".format(epoch+1, cfg.epochs, sum(train_loss)/len(train_loss)))
        
        # valid
        valid_loss = []
        with torch.no_grad():
            for i, (img, labels) in enumerate(test_loader):
                img, labels = img.to(cfg.device), labels.to(cfg.device)
                pred = model(img)
                loss = criterion(pred, labels)
                valid_loss.append(loss.item())

            average_loss = sum(valid_loss)/len(valid_loss)
            if average_loss < min_loss:
                min_loss = average_loss
                cfg.logger.info("##################saving model parameter##################")
                torch.save(model.state_dict(), cfg.save_model_name)

        cfg.logger.info("Epoch:[{0}/{1}] Validing Loss: {2}".format(epoch+1, cfg.epochs, average_loss))

        cfg.tensorboard_writer.add_scalars('Loss',{'train':sum(train_loss)/len(train_loss), 'valid':average_loss}, epoch+1)

        # print the current learnig rate
        cfg.logger.info("the current learning rate: {0}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        scheduler.step()
        

def keypoint_rcnn_train(cfg, model, optimizer, scheduler, train_loader, test_loader):
    min_oks = float('-inf')
    for epoch in range(cfg.epochs):
        # train
        total_train_loss = []
        train_loss_classifier = []
        train_loss_box_reg = []
        train_loss_keypoint = []
        train_loss_objectness = []
        train_loss_rpn_box_reg = []
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = list(image.to(f'cuda:{model.device_ids[0]}') for image in images)
            labels = [{k: v.to(f'cuda:{model.device_ids[0]}') for k, v in t.items()} for t in labels]

            # forward
            loss_dict, _ = model(images, labels)
            train_loss_classifier.append(loss_dict['loss_classifier'].item())
            train_loss_box_reg.append(loss_dict['loss_box_reg'].item())
            train_loss_keypoint.append(loss_dict['loss_keypoint'].item())
            train_loss_objectness.append(loss_dict['loss_objectness'].item())
            train_loss_rpn_box_reg.append(loss_dict['loss_rpn_box_reg'].item())
            
            losses = sum(loss for loss in loss_dict.values())

            total_train_loss.append(losses.item())

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # if i == 3 * cfg.print_interval:
            #     break

            if (i+1) % cfg.print_interval == 0:
                cfg.logger.info("Epoch:[{0}/{1}] Total Training Loss: [{2}]".format(epoch+1, cfg.epochs, sum(total_train_loss)/len(total_train_loss)))
                cfg.logger.info("loss_classifier:[{0}]  loss_box_reg:[{1}]  loss_keypoint:[{2}]  loss_objectness:[{3}]  loss_rpn_box_reg:[{4}]".format(
                    sum(train_loss_classifier)/len(train_loss_classifier),
                    sum(train_loss_box_reg)/len(train_loss_box_reg),
                    sum(train_loss_keypoint)/len(train_loss_keypoint),
                    sum(train_loss_objectness)/len(train_loss_objectness),
                    sum(train_loss_rpn_box_reg)/len(train_loss_rpn_box_reg),
                ))

        cfg.logger.info("Epoch:[{0}/{1}] Total Training Loss: [{2}]".format(epoch+1, cfg.epochs, sum(total_train_loss)/len(total_train_loss)))
        cfg.logger.info("loss_classifier:[{0}]  loss_box_reg:[{1}]  loss_keypoint:[{2}]  loss_objectness:[{3}]  loss_rpn_box_reg:[{4}]".format(
                    sum(train_loss_classifier)/len(train_loss_classifier),
                    sum(train_loss_box_reg)/len(train_loss_box_reg),
                    sum(train_loss_keypoint)/len(train_loss_keypoint),
                    sum(train_loss_objectness)/len(train_loss_objectness),
                    sum(train_loss_rpn_box_reg)/len(train_loss_rpn_box_reg),
                ))
        
        # valid
        model.eval()
        with torch.no_grad():
            total_oks = []
            for i, (images, labels) in enumerate(test_loader):
                images = list(image.to(f'cuda:{model.device_ids[0]}') for image in images)
                labels = [{k: v.to(f'cuda:{model.device_ids[0]}') for k, v in t.items()} for t in labels]

                # forward
                _, detections = model(images, labels)

                metrics = calculate_metrics(detections, labels)
                total_oks.append(metrics['oks'])

                if (i+1) % cfg.print_interval == 0:
                    cfg.logger.info("Epoch:[{0}/{1}] Total Validing OKS: [{2}]".format(epoch+1, cfg.epochs, sum(total_oks)/len(total_oks)))

            average_oks = sum(total_oks)/len(total_oks)
            if average_oks > min_oks:
                min_oks = average_oks
                cfg.logger.info("##################saving model parameter##################")
                torch.save(model.state_dict(), cfg.save_model_name)

            cfg.logger.info("Epoch:[{0}/{1}] Total Validing OKS: [{2}]".format(epoch+1, cfg.epochs, average_oks))
            
            cfg.tensorboard_writer.add_scalars(
                'Loss',{
                    'train_loss_classifier':sum(train_loss_classifier)/len(train_loss_classifier),
                    'train_loss_box_reg':sum(train_loss_box_reg)/len(train_loss_box_reg),
                    'train_loss_keypoint':sum(train_loss_keypoint)/len(train_loss_keypoint),
                    'train_loss_objectness':sum(train_loss_objectness)/len(train_loss_objectness),
                    'train_loss_rpn_box_reg':sum(train_loss_rpn_box_reg)/len(train_loss_rpn_box_reg),
                    # 'valid_loss_classifier':sum(valid_loss_classifier)/len(valid_loss_classifier),
                    # 'valid_loss_box_reg':sum(valid_loss_box_reg)/len(valid_loss_box_reg),
                    # 'valid_loss_keypoint':sum(valid_loss_keypoint)/len(valid_loss_keypoint),
                    # 'valid_loss_objectness':sum(valid_loss_objectness)/len(valid_loss_objectness),
                    # 'valid_loss_rpn_box_reg':sum(valid_loss_rpn_box_reg)/len(valid_loss_rpn_box_reg),
                    'total_train_loss':sum(total_train_loss)/len(total_train_loss),
                    'total_valid_oks':average_oks,

                },
                epoch+1
            )

            # print the current learnig rate
            cfg.logger.info("the current learning rate: {0}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        scheduler.step()


def keypoint_rcnn_finetune_train(cfg, model, optimizer, scheduler, train_loader, test_loader):
    pass


def PandaRobotTrain():
    cfg = set_config(False)

    cfg.logger.info("############## the network type is [{0}] ##############".format(cfg.network_type))

    rds = robotDataSets(cfg.data_path, cfg.robot_type, cfg.network_type)
    rds_num_data = len(rds)
    print("the num of image:", rds_num_data)
    n_train_data = int(round(rds_num_data * cfg.training_data_proportion))
    n_valid_data = rds_num_data - n_train_data
    rds_train, rds_test = torch.utils.data.random_split(rds, [n_train_data, n_valid_data])

    train_loader = DataLoader(dataset=rds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_worker, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=rds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_worker, collate_fn=collate_fn)

    model = MyModel(cfg)


    if cfg.network_type == 'resnet':
        criterion = torch.nn.MSELoss()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=cfg.lr_scheduler_step_size, gamma=cfg.lr_scheduler_gamma)

        if cfg.device.type == 'cuda':
            torch.cuda.set_device(cfg.gpu_id)
            model.cuda()
            criterion.cuda()

        cfg.logger.info("Training started!")

        resnet50_train(cfg=cfg,
                    model=model,
                    criterion=criterion, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    train_loader=train_loader,
                    test_loader=test_loader,)
        
        cfg.logger.info("Training finished!")
    

    elif cfg.network_type == 'keypoint_rcnn':
        params = [p for p in model.parameters() if p.requires_grad]
        # optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=0.0005)
        optimizer = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=cfg.lr_scheduler_step_size, gamma=cfg.lr_scheduler_gamma)
        
        if cfg.device.type == 'cuda':
            torch.cuda.set_device(cfg.gpu_id)
            model.cuda()

        cfg.logger.info("Training started!")

        keypoint_rcnn_train(cfg=cfg,
                    model=model,
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    train_loader=train_loader,
                    test_loader=test_loader,)
        
        cfg.logger.info("Training finished!")


    elif cfg.network_type == 'keypoint_rcnn_finetune':
        params = [p for p in model.parameters() if p.requires_grad]
        # optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=0.0005)
        optimizer = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=cfg.lr_scheduler_step_size, gamma=cfg.lr_scheduler_gamma)

        if cfg.device.type == 'cuda':
            if len(cfg.gpu_id) == 1:
                torch.cuda.set_device(cfg.gpu_id[0])
                model.cuda()
            else:
                model = nn.DataParallel(model, device_ids=[gpu for gpu in cfg.gpu_id])
                model.to(f'cuda:{model.device_ids[0]}')

        cfg.logger.info("Training started!")

        keypoint_rcnn_train(cfg=cfg,
                    model=model,
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    train_loader=train_loader,
                    test_loader=test_loader)
        
        cfg.logger.info("Training finished!")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument(
    #     "-i", "--data-path", 
    #     # required=True, 
    #     help="the path of training data."
    # )

    # args = parser.parse_args()

    PandaRobotTrain()
