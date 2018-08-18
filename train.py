import argparse
from utils.utils import *
from utils.line import Line
from tqdm import trange
import torch
import torch.optim as optim
import sys
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_path", type=str)
    parser.add_argument("-save", "--save_path", type=str)
    parser.add_argument("-lossdata", "--lossdata_path", type=str)

    # Hyperparams.
    parser.add_argument("-order", "--order", type=int, default=2)
    parser.add_argument("-neg", "--negsamplesize", type=int, default=5)
    parser.add_argument("-dim", "--dimension", type=int, default=128)
    parser.add_argument("-batchsize", "--batchsize", type=int, default=5)
    parser.add_argument("-epochs", "--epochs", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=0.025)  # As starting value in paper
    parser.add_argument("-negpow", "--negativepower", type=float, default=0.75)
    args = parser.parse_args()

    # Create dict of distribution when opening file
    edgedistdict, nodedistdict, weights, nodedegrees, maxindex = makeDist(
        args.graph_path, args.negativepower)

    edgesaliassampler = VoseAlias(edgedistdict)
    nodesaliassampler = VoseAlias(nodedistdict)

    batchrange = int(len(edgedistdict) / args.batchsize)
    print(maxindex)
    line = Line(maxindex + 1, embed_dim=args.dimension, order=args.order)

    opt = optim.SGD(line.parameters(), lr=args.learning_rate,
                    momentum=0.9, nesterov=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lossdata = {"it": [], "loss": []}
    it = 0

    print("\nTraining on {}...\n".format(device))
    for epoch in range(args.epochs):
        print("Epoch {}".format(epoch))
        for b in trange(batchrange):
            samplededges = edgesaliassampler.sample_n(args.batchsize)
            batch = list(makeData(samplededges, args.negsamplesize, weights, nodedegrees,
                                  nodesaliassampler))
            batch = torch.LongTensor(batch)
            v_i = batch[:, 0]
            v_j = batch[:, 1]
            negsamples = batch[:, 2:]
            line.zero_grad()
            loss = line(v_i, v_j, negsamples, device)
            loss.backward()
            opt.step()

            lossdata["loss"].append(loss.item())
            lossdata["it"].append(it)
            it += 1

    print("\nDone training, saving model to {}".format(args.save_path))
    torch.save(line, "{}".format(args.save_path))

    print("Saving loss data at {}".format(args.lossdata_path))
    with open(args.lossdata_path, "wb") as ldata:
        pickle.dump(lossdata, ldata)
    sys.exit()
