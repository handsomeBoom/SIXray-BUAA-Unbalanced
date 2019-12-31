import sys
import os
import cv2
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import SIXray_ROOT, SIXray_CLASSES as labelmap
from PIL import Image
from data import SIXrayAnnotationTransform, SIXrayDetection, BaseTransform, SIXray_CLASSES
import torch.utils.data as data
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/model-1231.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='../predicted_file/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--sixray_root', default=SIXray_ROOT, help='Location of SIXray root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    file_set = save_folder+'file_set.txt'
    core_annotation = save_folder + 'det_test_带电芯充电宝.txt'
    coreless_annotation = save_folder + 'det_test_不带电芯充电宝.txt'

    if os.path.exists(core_annotation):
        os.remove(core_annotation)
    if os.path.exists(coreless_annotation):
        os.remove(coreless_annotation)
    if os.path.exists(file_set):
        os.remove(file_set)
    num_images = len(testset)
    
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        image_name = os.path.basename(img_id).split('.')[0]
        
        with open(file_set, mode='a+') as f:
            f.write(image_name + '\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data

        im_det = img.copy()
        h, w, _ = im_det.shape
        # scale each detection back up to the image
        scale = torch.Tensor([w, h, w, h])


        need_save = False
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.01:
                b = detections[0, i, j, 0].item()

                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = [ n for n in pt]
                
                name_words = labelmap[i-1]

                if name_words[0] == '带':
                    with open(core_annotation, 'a+') as f:
                        f.write(image_name + ' ' + str(b) + ' ' + str(coords[0]) + ' ' + str(coords[1]) + ' ' + 
                                str(coords[2])+' '+str(coords[3])+'\n');
                    name_words = 'Battery_Core'
                else:
                    with open(coreless_annotation, 'a+') as f:
                        f.write(image_name + ' ' + str(b) + ' ' + str(coords[0]) + ' ' + str(coords[1]) + ' ' + 
                                str(coords[2]) + ' ' + str(coords[3]) + '\n');
                    name_words = 'Battery_Coreless'

                #if b >= 0.2:
                #    cv2.rectangle(im_det, (coords[0], coords[1]), (coords[2], coords[3]), (0, 225, 0), 2)
                #    cv2.putText(im_det, name_words, (coords[0], coords[1] - 5), 0, 0.6, (0, 225, 0), 2)

                need_save = False
                j += 1

        if need_save:
            dst_path = img_id.replace('Image', 'ImageTarget')
            dst_dir = os.path.dirname(dst_path)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            cv2.imwrite(dst_path, im_det)

def test(img_path, anno_path):
    num_classes = len(SIXray_CLASSES) + 1
    net = build_ssd('test', 300, num_classes)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print(" Model loaded")
    testset = SIXrayDetection(img_path, anno_path, target_transform=SIXrayAnnotationTransform()) 
    print("test set Initialed")

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    test_net(args.save_folder, net, args.cuda, testset, 
                BaseTransform(net.size, (104, 117, 123)),
                thresh=args.visual_threshold)

if __name__ == '__main__':
    test('', '')
