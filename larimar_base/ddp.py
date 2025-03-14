import logging
import os
import subprocess
import numpy as np

import torch.distributed as dist


def get_nccl_socket_ifname():
    ipa = subprocess.run(['ip', 'a'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ipa.stdout.decode('utf-8').split('\n')
    all_names = []
    name = None
    for line in lines:
        if line and not line[0] == ' ':
            name = line.split(':')[1].strip()
            continue
        if 'link/infiniband' in line:
            all_names.append(name)
    os.environ['NCCL_SOCKET_IFNAME'] = ','.join(all_names)


def fix_infiniband():
    # os.environ['NCCL_SOCKET_IFNAME'] = "^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp,bond"

    # ifname = os.environ.get('NCCL_SOCKET_IFNAME', None)
    # if ifname is None:
    #     os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker0'
    get_nccl_socket_ifname()
    os.environ['NCCL_IB_CUDA_SUPPORT'] = '1'
    ibv = subprocess.run('ibv_devinfo', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode('utf-8').split('\n')
    exclude = ''
    include = ''
    for line in lines:
        if 'hca_id:' in line:
            name = line.split(':')[1].strip()
        if '\tport:' in line:
            port = line.split(':')[1].strip()
        if 'link_layer:' in line and 'Ethernet' in line:
            exclude = exclude + f'{name}:{port},'
        if 'link_layer:' in line and 'infiniband' in line.lower():
            include = include + f'{name}:{port},'
    if exclude:
        exclude = '^' + exclude[:-1]
        # print(exclude)
        os.environ['NCCL_IB_HCA'] = exclude
    else:
        os.environ['NCCL_IB_HCA'] = include[:-1]



fix_inifiniband = fix_infiniband  # For backwards compatibility

def init_ddp_process_group(local_rank: int = None, port: int = None, world_size: int = None, dist_rank: int = None,
                           overwrite_env_vars=True):
    logger = logging.getLogger('InitDDP')
    if os.environ.get('LSB_JOBID', False):
        local_rank = int(os.environ.get('LSF_PM_XPROCID', 1)) - 1 if local_rank is None else local_rank

        hostname = os.environ.get('HOSTNAME', 'localhost')
        num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
        node_rank = int(os.environ.get('LSF_PM_XMACHID', 1)) - 1
        dist_rank = node_rank * num_gpus + local_rank if dist_rank is None else dist
        num_hosts = len(os.environ.get('LSB_MCPU_HOSTS', 'localhost cpus').split()) // 2
        rng = np.random.RandomState(seed=int(os.environ.get('LSB_JOBID', 0)))
        master_host = os.environ.get('LSF_FROM_HOST', 'localhost')
        port = rng.randint(10000, 20000) if port is None else port
        if num_hosts > 1:
            fix_inifiniband()
        prefix = f'{hostname}, Local Rank {local_rank}/{num_gpus}, Global Rank {dist_rank}/{world_size}:'

        logger.info(f'{prefix} Trying to init process group')
        logger.debug(f'{prefix} CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
        logger.debug(f'{prefix} LSF_PM_XMACHID=', os.environ.get('LSF_PM_XMACHID', ''))
        logger.debug(f'{prefix} LSF_PM_XPROCID=', os.environ.get('LSF_PM_XPROCID', ''))
        logger.debug(f'{prefix} LSB_MCPU_HOSTS=', os.environ.get('LSB_MCPU_HOSTS', ''))
        logger.debug(f'{prefix} MASTER_ADDR=', master_host)
        logger.debug(f'{prefix} MASTER_PORT=', port)
    elif os.environ.get('SLURM_JOB_ID', False):

        num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
        local_rank = int(os.environ.get('SLURM_PROCID', 0)) % num_gpus if local_rank is None else local_rank
        node_rank = int(os.environ.get('SLURM_NODEID', 0))

        hostlist = subprocess.run(['scontrol', 'show', 'hostnames', os.environ.get('SLURM_JOB_NODELIST', 'localhost')],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        hostlist = hostlist.stdout.decode('utf8').strip().split('\n')
        num_hosts = len(hostlist)
        master_host = hostlist[0]
        hostname = os.environ.get('HOSTNAME', 'localhost')
        dist_rank = node_rank * num_gpus + local_rank if dist_rank is None else dist
        rng = np.random.RandomState(seed=int(os.environ.get('SLURM_JOB_ID', 0)))
        port = rng.randint(10000, 20000) if port is None else port

        prefix = f'{hostname}, Local Rank {local_rank}/{num_gpus}, Global Rank {dist_rank}/{world_size}:'

        logger.info(f'{prefix} Trying to init process group')
        logger.debug(f'{prefix} CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
        logger.debug(f'{prefix} SLURM_NODEID=', os.environ.get('SLURM_NODEID', ''))
        logger.debug(f'{prefix} SLURM_PROCID=', os.environ.get('SLURM_PROCID', ''))
        logger.debug(f'{prefix} SLURM_JOB_NODELIST=', os.environ.get('SLURM_JOB_NODELIST', ''))
        logger.debug(f'{prefix} MASTER_ADDR=', master_host)
        logger.debug(f'{prefix} MASTER_PORT=', port)

    else:
        return dist.init_process_group(backend='nccl', init_method='env://')
    world_size = num_gpus * num_hosts if world_size is None else world_size

    if 'RANK' not in os.environ.keys() or overwrite_env_vars:
        os.environ['RANK'] = str(dist_rank)
    if 'LOCAL_RANK' not in os.environ.keys() or overwrite_env_vars:
        os.environ['LOCAL_RANK'] = str(local_rank)
    if 'NODE_RANK' not in os.environ.keys() or overwrite_env_vars:
        os.environ['NODE_RANK'] = str(node_rank)
    if 'MASTER_ADDR' not in os.environ.keys() or overwrite_env_vars:
        os.environ['MASTER_ADDR'] = master_host
    if 'WORLD_SIZE' not in os.environ.keys() or overwrite_env_vars:
        os.environ['WORLD_SIZE'] = str(world_size)

    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' not in os.environ.keys() or overwrite_env_vars:

        os.environ['MASTER_PORT'] = str(port)


    group = dist.init_process_group(backend='nccl', init_method='env://')

    logger.info(f'{prefix} Done init process group')
    return group
