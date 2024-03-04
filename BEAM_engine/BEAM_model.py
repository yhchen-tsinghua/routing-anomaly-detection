import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
import pickle
import time
from itertools import cycle
from pathlib import Path

class Analyzer(torch.utils.data.Dataset):

    def __init__(self, Q):
        super(Analyzer, self).__init__()
        self.Q = Q
        self.edge_list = []

        self.asn_list = []
        self.asn2idx = {}

        self.downstreams = []
        self.upstreams = []

    def read_edge_file(self, edge_file):
        def get_index(asn):
            if asn not in self.asn2idx:
                self.asn2idx[asn] = len(self.asn_list)
                self.asn_list.append(asn)
                self.downstreams.append(set())
                self.upstreams.append(set())
            return self.asn2idx[asn]

        for line in open(edge_file, "r"):
            if line[0] == "#": continue
            i, j, k = line.strip().split("|")[:3]

            index_i = get_index(i)
            index_j = get_index(j)

            assert index_i != index_j

            if k == "0":
                self.edge_list.append((index_i, index_j))
                self.downstreams[index_i].add(index_j)
                self.upstreams[index_j].add(index_i)

                self.edge_list.append((index_j, index_i))
                self.downstreams[index_j].add(index_i)
                self.upstreams[index_i].add(index_j)
            elif k == "-1":
                self.edge_list.append((index_i, index_j))
                self.downstreams[index_i].add(index_j)
                self.upstreams[index_j].add(index_i)
            else:
                raise RuntimeError(f"unexpected rel {rel}")

        print(f"nodes: {len(self.asn_list)}")
        print(f"edges: {len(self.edge_list)}")

        self.init_sample_method()

        return self

    def init_sample_method(self, eps=0.01):
        upstreams = self.upstreams
        downstreams = self.downstreams

        global_cycler = cycle(range(len(self.asn_list)))
        none_cycler = cycle([None])

        # providers as tails, thus negative samples
        negative_tails = [cycle(u-d) if u-d else none_cycler
                            for u,d in zip(upstreams, downstreams)]

        # customers as heads, thus negative samples
        negative_heads = [cycle(d-u) if d-u else none_cycler
                            for u,d in zip(upstreams, downstreams)]

        def get_local_tail_negative(head):
            return next(negative_tails[head])

        def get_local_head_negative(tail):
            return next(negative_heads[tail])

        def get_global_tail_negative(head):
            for tail_negative in global_cycler:
                if tail_negative != head and tail_negative not in downstreams[head]:
                    return head, tail_negative

        def get_global_head_negative(tail):
            for head_negative in global_cycler:
                if head_negative != tail and head_negative not in upstreams[tail]:
                    return head_negative, tail

        bound1 = 0.5-eps
        bound2 = 0.5+eps

        def draw_negative_sample(head, tail):
            r = np.random.random()
            if r < bound1: # try corrupt tail
                tail_negative = get_local_tail_negative(head)
                if tail_negative:
                    sample = (head, tail_negative)
                else: # try corrupt head
                    head_negative = get_local_head_negative(tail)
                    if head_negative:
                        sample = (head_negative, tail)
                    else: # global negative sample
                        sample = get_global_tail_negative(head)

            elif r > bound2: # try corrupt head
                head_negative = get_local_head_negative(tail)
                if head_negative:
                    sample = (head_negative, tail)
                else: # try corrupt tail
                    tail_negative = get_local_tail_negative(head)
                    if tail_negative:
                        sample = (head, tail_negative)
                    else: # global negative sample
                        sample = get_global_head_negative(tail)

            else: # global negative sample
                if r < 0.5:
                    sample = get_global_head_negative(tail)
                else:
                    sample = get_global_tail_negative(head)

            return sample

        self.draw_negative_sample = draw_negative_sample

    def __len__(self):
        return len(self.edge_list) * self.Q

    def __getitem__(self, index):
        positive_sample = self.edge_list[index // self.Q]
        negative_sample = self.draw_negative_sample(*positive_sample)
        input_vector = [0, *positive_sample, *negative_sample]
        return torch.tensor(input_vector, dtype=torch.int64, requires_grad=False)


class BEAM(torch.nn.Module):

    def __init__(self, edge_file, Q=5, dimension=128, train_dir=Path("./"), cuda_device='cuda', num_workers=20):
        super(BEAM, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(cuda_device if self.use_cuda else 'cpu')
        print("device: {}".format(self.device))

        self.train_dir = train_dir

        self.analyzer = Analyzer(Q).read_edge_file(edge_file)
        self.node_embedding = torch.nn.Embedding(
                                len(self.analyzer.asn_list), dimension)
        self.rela_embedding = torch.nn.Embedding(1, dimension)
        self.link_embedding = torch.nn.Embedding(1, dimension)

        self.num_workers = num_workers

    def forward(self, batchVector):
        idx_k = batchVector[:,0]
        link = torch.nn.functional.softmax(self.link_embedding(idx_k), dim=1)
        rela = self.rela_embedding(idx_k)
        pi = self.node_embedding(batchVector[:,1])
        pj = self.node_embedding(batchVector[:,2])
        ni = self.node_embedding(batchVector[:,3])
        nj = self.node_embedding(batchVector[:,4])

        # softplus(corrupt - correct)
        relaError = torch.sum((nj-ni-pj+pi)*rela, dim=1) # criteria 2
        linkError = torch.sum((pj-pi-nj+ni)*(pj-pi+nj-ni)*link, dim=1)
        loss = torch.nn.functional.softplus(relaError + linkError)

        return loss

    def train(self, epoches=500):
        self.to(device=self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)
        generator = torch.utils.data.DataLoader(
                        self.analyzer, batch_size=1024, shuffle=True,
                        num_workers=self.num_workers)

        for epoch in range(1, epoches + 1):
            if epoch%100 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, self.train_dir / "checkpoint")
            loss = 0.0
            tik = time.time()
            for batchData in generator:
                optimizer.zero_grad()
                batchData = batchData.to(device=self.device)
                batchLoss = self(batchData).sum()
                loss += float(batchLoss)
                batchLoss.backward()
                optimizer.step()
            tok = time.time()
            print(f"Epoch{epoch:4d}/{epoches} Loss: {loss:e} Time: {tok-tik:.1f}s")

    def save_embeddings(self, path='.'):
        print("save embeddings...")
        path = Path(path)
        node_keys = self.analyzer.asn_list
        rela_keys = ["p2c"]
        link_keys = ["p2c"]

        def dump_embedding(keys, tensor, filePath):
            if self.use_cuda:
                emb = tensor.weight.cpu().data.numpy()
            else:
                emb = tensor.weight.data.numpy()
            emb = dict(zip(keys, emb))
            pickle.dump(emb, open(filePath, 'wb'))

        dump_embedding(node_keys, self.node_embedding, path/'node.emb')
        dump_embedding(rela_keys, self.rela_embedding, path/'rela.emb')
        dump_embedding(link_keys, self.link_embedding, path/'link.emb')
