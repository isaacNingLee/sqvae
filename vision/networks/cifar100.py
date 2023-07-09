from networks.net_32 import EncoderVqResnet32, DecoderVqResnet32
from networks.pixel_cnn import PixelCNN


class EncoderVq_resnet(EncoderVqResnet32):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "Cifar100"


# class DecoderVq_resnet(DecoderVqResnet32):
#     def __init__(self, dim_z, cfgs, flg_bn):
#         super(DecoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn)
#         self.dataset = "Cifar100"

class DecoderVq_resnet(PixelCNN)):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(PixelCNN, self).__init__(n_chanels=dim_z.shape[1], n_layers=7, img_shape=cfgs.dataset.shape)
        self.dataset = "Cifar100"