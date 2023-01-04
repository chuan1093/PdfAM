import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from time import time
from network.chooseModel import *
import subprocess
from utils import *

parser = ArgumentParser(description='')


# parser.add_argument('--mark_str', type=str, default='Brain_Radial_ISTARB', help='result directory')
parser.add_argument('--mark_str', type=str, default='Brain_Radial_ISTARB_PdfAMGau', help='result directory')
# parser.add_argument('--mark_str', type=str, default='Brain_Radial_ISTARB_PdfAMGmmT3K2', help='result directory')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=1000, help='epoch number of end training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--batch_size', type=int, default=4, help='')
parser.add_argument('--cs_ratio', type=int, default=5, help='ratio of sample mask')
parser.add_argument('--save_interval', type=int, default=10, help='eg, save model every 10 epochs')

parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

parser.add_argument('--trainset_name', type=str, default='brain_train_256x256_100.mat', help='train dataset')
parser.add_argument('--trainset_name_matkey', type=str, default='labels', help='')
parser.add_argument('--mask_path', type=str, default='sampling_matrix/mask_radial_5.mat', help='sampling matrix directory')
parser.add_argument('--mask_path_matkey', type=str, default='mask_matrix', help='')

# parser.add_argument('--model_type', type=str, default='ISTARB', help='')
parser.add_argument('--model_type', type=str, default='ISTARB_PdfAMGau', help='')
# parser.add_argument('--model_type', type=str, default='ISTARB_PdfAMGmm', help='')

parser.add_argument('--optimizer_type', type=str, default='Adam', help='')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='')
parser.add_argument('--show_model_paras', type=int, default=0, help='')


parser.add_argument('--layer_num', type=int, default=9, help='phase number')
parser.add_argument('--rb_num', type=int, default=2, help='')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--mod', type=int, default=0, help='')
parser.add_argument('--norm', type=int, default=0, help='')
# for PdfAM-Gmm
parser.add_argument('--gmm_num_T', type=int, default=3, help='')
parser.add_argument('--gmm_num_K', type=int, default=2, help='')

# for test_while_train
parser.add_argument('--test_while_train', type=int, default=1, help='whether to test when saving model')
parser.add_argument('--test_name', type=str, default='brain_test_50', help='name of test set')
parser.add_argument('--test_img_type', type=str, default='png', help='name of test set')


args = parser.parse_args()

# device cuda
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
Training_data_Name = args.trainset_name
mask_path = args.mask_path
model_type = args.model_type
optimizer_type = args.optimizer_type
batch_size = args.batch_size

mod = (args.mod == 1)
norm = (args.norm == 1)
test_while_train = (args.test_while_train == 1)

model_full_name = '%s_ratio_%d_layer_%d_group_%d' % (args.mark_str, cs_ratio, layer_num, group_num)
model_dir = os.path.join(args.model_dir, model_full_name)
log_file_name = "./%s/Log_TRAIN_%s.txt" % (args.log_dir, model_full_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# load mask
mask_suffix = mask_path.split('.')[-1]
if mask_suffix == 'mat':
    mask_np = sio.loadmat(mask_path)[args.mask_path_matkey]
elif mask_suffix == 'npy':
    mask_np = np.load(mask_path)
else:
    assert False, "mask supports .mat and .npy only!"
mask_torch = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)
mask_torch = mask_torch.unsqueeze(-1).unsqueeze(0)

# load data
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data[args.trainset_name_matkey]
nrtrain = Training_labels.shape[0]

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

model = chooseModel(args)
model = nn.DataParallel(model)
model = model.to(device)
show_model_paras(model, args.show_model_paras==1)
optimizer = get_optimizer(args.optimizer_type, model.parameters(), learning_rate, args.weight_decay)


if start_epoch == -1:
    pre_model_dir = model_dir
    import glob
    filelist = sorted(glob.glob('%s/net_params_*.pkl' % pre_model_dir))
    if len(filelist) == 0:
        start_epoch = 0
        print('start epoch is -1, i.e., starting from epoch %d' % start_epoch)
    else:
        int_list = []
        for i in range(len(filelist)):
            model_path = filelist[i]
            this_epoch = int(os.path.split(model_path)[-1].split('.')[0].split('_')[-1])
            int_list.append(this_epoch)
        start_epoch = max(int_list)
        print('start epoch is -1, i.e., starting from epoch %d' % start_epoch)
if start_epoch > 0:
    pre_model_dir = model_dir
    model_path = '%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

TRAIN_TIME_All = np.zeros([1, (end_epoch - start_epoch)], dtype=np.float32)
TRAIN_TIME_All_i = 0
for epoch_i in range(start_epoch+1, end_epoch+1):
    batch_cnt = 0
    loss_all_accum = 0
    loss_discrepancy_accum = 0
    train_start = time()
    model.train()
    for data in rand_loader:
        gt = data.to(device)
        gt = gt.view(gt.shape[0], 1, gt.shape[1], gt.shape[2])

        xu_real = zero_filled_pt190(gt, mask_torch, mod=mod, norm=norm)

        model_output = model(xu_real, mask_torch)

        x_output = model_output

        loss_all = torch.mean(torch.pow(x_output - gt, 2))

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        batch_cnt += 1
        loss_all_accum += loss_all.item()

    train_end = time()
    train_epoch_time = train_end - train_start
    TRAIN_TIME_All[0, TRAIN_TIME_All_i] = train_epoch_time
    TRAIN_TIME_All_i += 1

    output_data = "[%02d/%02d] Total Loss: %.5f, Train Time: %.4f \n" % (
        epoch_i, end_epoch, loss_all_accum / batch_cnt, train_epoch_time)
    print(output_data, end='')
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))
        if test_while_train:
            cmd_png = 'python test.py --epoch_num %d --layer_num %d --group_num %d --cs_ratio %d --model_type %s' \
                      ' --mask_path %s --mask_path_matkey %s --test_name %s --test_img_type %s --mark_str %s' \
                      ' --mod %d --norm %d --rb_num %d '  % (
                          epoch_i, layer_num, group_num, cs_ratio, model_type,
                          mask_path, args.mask_path_matkey, args.test_name, args.test_img_type, args.mark_str,
                          args.mod, args.norm, args.rb_num)
            print(cmd_png, end='\n')
            process_png = subprocess.Popen(cmd_png, shell=True)
            process_png.wait()

output_data = 'From epoch %d to epoch %d, Average Train Epoch Time: %.4f \n' % (start_epoch, end_epoch, np.mean(TRAIN_TIME_All))
output_file = open(log_file_name, 'a')
output_file.write(output_data)
output_file.close()