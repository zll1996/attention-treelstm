import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees')
    parser.add_argument('--name', default='default_name',
                        help='name for log and saved models')
    parser.add_argument('--saved', default='saved_model',
                        help='name for log and saved models')

    parser.add_argument('--model_name', default='constituency',
                        help='model name constituency or dependency')
    parser.add_argument('--data', default='data/sst/',
                        help='path to dataset')
    parser.add_argument('--glove', default='data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--batchsize', default=25, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.05, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--emblr', default=0.1, type=float,
                        metavar='EMLR', help='initial embedding learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--reg', default=1e-4, type=float,
                        help='l2 regularization (default: 1e-4)')
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    parser.add_argument('--fine_grain', default=0, type=int,
                        help='fine grained (default 0 - binary mode)')
    parser.add_argument('--input_dim', default=300, type=int,
                        help='dimension size of input layer(default: 300)')
    parser.add_argument('--attention_dim', default=350, type=int,
                        help='dimension size of attention layer(default: 200)')
    parser.add_argument('--attention_flag', default='True', type=bool,
                        help='add attention layer or not(default: False)')
    parser.add_argument('--dropout2', default=0.5, type=float,
                        help='dropout for attention layer(default: 0.5)')

                        # untest on fine_grain yet.
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    cuda_parser.add_argument('--lower', dest='cuda', action='store_true')
    parser.set_defaults(cuda=True)
    parser.set_defaults(lower=True)

    args = parser.parse_args()
    return args
