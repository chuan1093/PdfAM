import scipy.io as sio
import os
import glob
from time import time
import cv2
from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
from network.chooseModel import *

parser = ArgumentParser(description='')

# parser.add_argument('--mark_str', type=str, default='Brain_Radial_ISTARB', help='result directory')
parser.add_argument('--mark_str', type=str, default='Brain_Radial_ISTARB_PdfAMGau', help='result directory')
# parser.add_argument('--mark_str', type=str, default='Brain_Radial_ISTARB_PdfAMGmmT3K2', help='result directory')

parser.add_argument('--epoch_num', type=int, default=530, help='epoch number of model')

parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--cs_ratio', type=int, default=5, help='ratio of sample mask')

parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

parser.add_argument('--test_name', type=str, default='brain_test_50', help='name of test set')
parser.add_argument('--test_img_type', type=str, default='png', help='name of test set')
parser.add_argument('--saveimg', type=int, default=1, help='')

parser.add_argument('--mask_path', type=str, default='sampling_matrix/mask_radial_5.mat', help='sampling matrix directory')
parser.add_argument('--mask_path_matkey', type=str, default='mask_matrix', help='')

# parser.add_argument('--model_type', type=str, default='ISTARB', help='')
parser.add_argument('--model_type', type=str, default='ISTARB_PdfAMGau', help='')
# parser.add_argument('--model_type', type=str, default='ISTARB_PdfAMGmm', help='')

parser.add_argument('--layer_num', type=int, default=9, help='phase number of KAN')
parser.add_argument('--rb_num', type=int, default=2, help='')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--mod', type=int, default=0, help='')
parser.add_argument('--norm', type=int, default=0, help='')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch_num = args.epoch_num
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
test_name = args.test_name
mask_path = args.mask_path
model_type = args.model_type

mod = (args.mod == 1)
norm = (args.norm == 1)

model_full_name = '%s_ratio_%d_layer_%d_group_%d' % (args.mark_str, cs_ratio, layer_num, group_num)
model_dir = os.path.join(args.model_dir, model_full_name)
res_file_name = "./%s/Log_TEST_%s_teston_%s.txt" % (args.log_dir, model_full_name, test_name)
result_dir = os.path.join(args.result_dir, model_full_name + '_teston_%s' % test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# load data
test_dir = os.path.join(args.data_dir, test_name)
assert args.test_img_type == 'png' or args.test_img_type == 'tif', "not support such test_img_type"
filepaths = glob.glob(test_dir + '/*.' + args.test_img_type)
ImgNum = len(filepaths)

# load mask
mask_suffix = mask_path.split('.')[-1]
if mask_suffix == 'mat':
    mask_np = sio.loadmat(mask_path)[args.mask_path_matkey]
elif mask_suffix == 'npy':
    mask_np = np.load(mask_path)
else:
    assert False, "mask supports .mat and .npy only!"
mask_torch = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)
mask_torch = mask_torch.unsqueeze(-1).unsqueeze(0)  # 1,H,W,1

model = chooseModel(args)
model = nn.DataParallel(model)
model = model.to(device)

# Load pre-trained model with epoch number
model_path = '%s/net_params_%d.pkl' % (model_dir, epoch_num)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
TEST_TIME_All = np.zeros([1, ImgNum], dtype=np.float32)

print('\n')
print("MRI CS Reconstruction Start")
model.eval()
with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]
        Iorg = cv2.imread(imgName, 0)
        Icol = Iorg.reshape(1, 1, 256, 256) / 255.0

        gt = torch.from_numpy(Icol)
        gt = gt.type(torch.FloatTensor).to(device)

        test_start = time()

        xu_real = zero_filled(gt, mask_torch, mod=mod, norm=norm)

        model_output = model(xu_real, mask_torch)

        test_end = time()

        if isinstance(model_output, list):
            x_output = model_output[0]
        else:
            x_output = model_output

        Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)
        X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)

        rec_PSNR = psnr(X_rec*255., Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255., Iorg.astype(np.float64), data_range=255)
        rec_time = test_end - test_start

        print("[%02d/%02d] Run time for %s is %.4f, Proposed PSNR is %.2f, Proposed SSIM is %.4f" % (
            img_no, ImgNum, imgName, rec_time, rec_PSNR, rec_SSIM))
        if args.saveimg == 1:
            im_rec_rgb = np.clip(X_rec*255, 0, 255).astype(np.uint8)
            imgname_split = os.path.split(imgName)[-1]
            resultName = "%s_ratio_%d_epoch_%d_%s_PSNR_%.2f_SSIM_%.4f.png" \
                         % (imgname_split, cs_ratio, epoch_num, model_full_name, rec_PSNR, rec_SSIM)
            savepath = os.path.join(result_dir, resultName)
            cv2.imwrite(savepath, im_rec_rgb)
        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM
        TEST_TIME_All[0, img_no] = rec_time

print('\n')
output_data = "CS ratio is %d, Avg Proposed PSNR/SSIM for %s is %.2f/%.4f, Avg Rec Time is %.4f, Epoch number of model is %d \n" % \
              (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(TEST_TIME_All), epoch_num)
print(output_data)
output_file = open(res_file_name, 'a')
output_file.write(output_data)
output_file.close()

print("MRI CS Reconstruction End")