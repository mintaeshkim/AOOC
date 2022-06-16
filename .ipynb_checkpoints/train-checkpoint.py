from model import OGNet
from opts import parse_opts
from dataloader import load_data_train

if __name__=="__main__":

    opt = parse_opts()
    train_loader = load_data_train(opt)
    model = OGNet(opt, train_loader)
    # model.cuda()
    model.train(opt.normal_class)