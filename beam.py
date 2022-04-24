import torch
import params
import operator
from queue import PriorityQueue
from model import create_pad_mask, get_seq_mask

class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, current_seq, logProb, length):
        self.prevNode = previousNode
        self.wordid = wordId
        self.current_seq = current_seq
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

def beam_decode(decoder, X_hidden, X_pad_mask, selected_K, selected_K_pad_mask, device):
    beam_width = 15
    topk = 1
    decoded_batch = []

    for idx in range(1):
        current_seq = torch.tensor([[]], dtype=torch.long, device=device)
        decoder_input = torch.tensor([[params.SOS]], dtype=torch.long, device=device)
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        node = BeamSearchNode(None, decoder_input.squeeze(0), current_seq, 0, 1)
        nodes = PriorityQueue()
        nodes.put((-node.eval(), node))
        qsize = 1

        while True:
            if qsize > 2000: break

            score, n = nodes.get()
            decoder_input = n.wordid
            y_input = torch.cat((n.current_seq, n.wordid.unsqueeze(0)), dim=1)
            tgt_mask = get_seq_mask(y_input.size(1)).to(device)
            tgt_pad_mask = create_pad_mask(y_input, params.PAD).to(device)

            if n.wordid.item() == params.EOS and n.prevNode != None:
                endnodes.append((score, n))
                if len(endnodes) >= number_required: break
                else: continue

            _, decoder_output = decoder(X_hidden, y_input, selected_K, X_pad_mask, tgt_pad_mask, selected_K_pad_mask, tgt_mask)
            log_prob, indexes = torch.topk(decoder_output[-1], beam_width, dim=1)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(n, decoded_t[0], y_input, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))

            qsize += len(nextnodes) - 1

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.item())
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.item())

            utterance = utterance[::-1]

        decoded_batch.append(utterance)

    return decoded_batch