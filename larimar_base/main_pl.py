import lightning
from lightning.pytorch.cli import LightningCLI
from lightning_model import MemNetLight
from lightning_data import DataModule
import os
import subprocess


def fix_infiniband():
    ibv = subprocess.run('ibv_devinfo', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode('utf-8').split('\n')
    exclude = ''
    for line in lines:
        if 'hca_id:' in line:
            name = line.split(':')[1].strip()
        if '\tport:' in line:
            port = line.split(':')[1].strip()
        if 'link_layer:' in line and 'Ethernet' in line:
            exclude = exclude + f'{name}:{port},'

    if exclude:
        exclude = '^' + exclude[:-1]
        print(exclude)
        os.environ['NCCL_IB_HCA'] = exclude


def set_env(master_port):
        LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(' ')  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
        HOST_LIST = LSB_MCPU_HOSTS[::2]  # Strips the cores per node items in the list
        os.environ["MASTER_ADDR"] = HOST_LIST[0]  # Sets the MasterNode to thefirst node on the list of hosts
        os.environ["MASTER_PORT"] = master_port
        os.environ["NODE_RANK"] = str(HOST_LIST.index(os.environ["HOSTNAME"]))  # Uses the list index for node rank, master node rank must be 0
        os.environ["NCCL_SOCKET_IFNAME"] = 'ib,bond' #"^docker0,lo"  # avoids using docker of loopback interface
        os.environ["NCCL_DEBUG"] = "INFO"  # sets NCCL debug to info, during distributed training, bugs in code show up as nccl errors
        os.environ["NCCL_IB_CUDA_SUPPORT"] = '1'  # Force use of infiniband


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.block_size", "data.block_size")
        parser.link_arguments("model.perturb", "data.perturb")
        parser.link_arguments("model.encoder_model_type", "data.encoder_model_type")
        parser.link_arguments("model.encoder_model_name_or_path", "data.encoder_model_name_or_path")
        parser.link_arguments("model.decoder_model_type", "data.decoder_model_type")
        parser.link_arguments("model.decoder_model_name_or_path", "data.decoder_model_name_or_path")
        parser.link_arguments("model.cache_dir", "data.cache_dir")
        parser.link_arguments("model.do_lower_case", "data.do_lower_case")


def cli_main():
    MyLightningCLI(model_class=MemNetLight, datamodule_class=DataModule, save_config_kwargs={"overwrite": True})
    #lightning.Trainer
    #lightning.pytorch.callbacks.ModelCheckpoint


if __name__ == "__main__":
    fix_infiniband()
    set_env('53108')
    cli_main()
