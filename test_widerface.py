from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVICES']="2"
import argparse
import paddle
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Retinaface')
# parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_epoch_70.pdparams',#Resnet50_Final.pth
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pdparams',#Resnet50_Final.pth
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    # if load_to_cpu:
    #     pretrained_dict = paddle.load(pretrained_path, map_location=lambda storage, loc: storage)
    # else:
    #     device = paddle.cuda.current_device()
    #     pretrained_dict = paddle.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    pretrained_dict = paddle.load(pretrained_path)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.set_state_dict(pretrained_dict)#, strict=False)
    return model

def infer(net, testset_folder, i, img_name, _t, target_size):
    image_path = testset_folder + img_name
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    # testing scale
    # target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if args.origin_size:
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = paddle.to_tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1) #paddle.transpose(img,perm=[2, 0, 1])
    img = paddle.to_tensor(img).unsqueeze(0)

    _t['forward_pass'].tic()
    loc, conf, landms = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    # priors = priors.to(device)
    prior_data = priors
    boxes = decode(loc.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    # boxes = boxes / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).cpu().numpy()[:, 1]
    landms = decode_landm(landms.squeeze(0), prior_data, cfg['variance'])
    scale1 = paddle.to_tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2]])
    # scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    # landms = landms / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    # order = paddle.argsort(scores)[::-1]
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    # dets = dets[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    _t['misc'].toc()

    # --------------------------------------------------------------------
    save_name = f"{args.save_folder}{target_size}/{img_name[:-4]}.txt"
    dirname = os.path.dirname(save_name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(save_name, "w") as fd:
        bboxs = dets
        file_name = os.path.basename(save_name)[:-4] + "\n"
        bboxs_num = str(len(bboxs)) + "\n"
        fd.write(file_name)
        fd.write(bboxs_num)
        for box in bboxs:
            x = int(box[0])
            y = int(box[1])
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            confidence = str(box[4])
            line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
            fd.write(line)

    print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

if __name__ == '__main__':
    paddle.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt" #"wider_val.txt" zjq

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    real_test_dataset_name = []
    for name in test_dataset:
        if name[-3:]=='jpg':
            real_test_dataset_name.append(name)
    num_images = len(real_test_dataset_name)
    # dataset = WiderFacetest(args.dataset_folder,args.origin_size)
    # num_images = len(dataset)
    # test_dataloader =paddle.io.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    arr_target_size = [500, 800, 1100, 1400, 1700]
    # arr_target_size = [500]
    # arr_target_size = [800]
    # arr_target_size = [1100]
    # arr_target_size = [1400]
    # arr_target_size = [1700]
    for target_size in arr_target_size:
        print(f"ts:{target_size}")
        for i, img_name in enumerate(real_test_dataset_name):
            infer(net, testset_folder, i, img_name, _t, target_size)
            

