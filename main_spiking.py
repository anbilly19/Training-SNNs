import argparse
import datetime
import os
from solver_spiking import Solver

def main(args):
    os.makedirs(args.model_path, exist_ok=True)
    
    solver = Solver(args)
    if args.train:
        solver.train()
        solver.test()
    elif args.test:
        solver.test()
    elif args.single_test:
        solver.single_batch_test(mode='single',df_name_weights = None,
                                 df_name_vloss=None,
                                 df_name_spikes=None
                                 )


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('-T', default=10, type=int, help='simulating time-steps')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--log_step', type=int, default=10)

    parser.add_argument('--dset', type=str, default='cifardvs', help=['mnist', 'fmnist','dvsg','cifar','cifardvs'])
    parser.add_argument("--img_size", type=int, default=128, help="Img size")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch Size")
    parser.add_argument("--n_channels", type=int, default=2, help="Number of channels")
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--model_path', type=str, default='./model')

    parser.add_argument("--embed_dim", type=int, default=84, help="dimensionality of the latent space")
    parser.add_argument("--n_attention_heads", type=int, default=12, help="number of heads to be used")
    parser.add_argument("--forward_mul", type=int, default=2, help="forward multiplier")
    parser.add_argument("--n_layers", type=int, default=5, help="number of encoder layers")
    parser.add_argument("--load_model", type=bool, default=False, help="Load saved model")
    parser.add_argument("--visualize", type=bool, default=False, help="visualize test data")
    parser.add_argument("--test", type=bool, default=False, help="test only")
    parser.add_argument("--train", type=bool, default=False, help="train and test")
    parser.add_argument("--single_test", type=bool, default=False, help="single test")
    parser.add_argument('--out_dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args.model_path = os.path.join(args.model_path, args.dset)
    print(args)
    
    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
