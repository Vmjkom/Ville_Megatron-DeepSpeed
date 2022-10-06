from torch.utils.tensorboard import SummaryWriter
import numpy as np

samples = {}
flops = {}

with open('samples.txt') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        uusi = lines[i].split()
        if i % 2 == 0:
            samples[float(uusi[0])] = float(uusi[1])
        else:
            flops[float(uusi[0])] = float(uusi[1])

writer = SummaryWriter(log_dir='logs/tb_logs',filename_suffix='averages')


samples_per_node = np.array(list(samples.keys()))
flops_per_node = np.array(list(flops.keys()))
nnodes = list(samples.values())
#Linear scaling calculated by multiplying the node (!)configurations with the samples per node list
linearity_samples = np.array(nnodes)*samples_per_node[0]
linearity_flops = np.array(nnodes)*flops_per_node[0]

for s,l,n in list(zip(samples_per_node,linearity_samples,nnodes)):
    writer.add_scalars('Average_samples',{'Samples_per_node': s,
                    'Linear': l}
                    ,n)

for f,l,n in list(zip(flops_per_node,linearity_flops,nnodes)):
    writer.add_scalars('Average_flops',{'Flops_per_node': f,
                    'Linear': l}
                    ,n)

writer.close()