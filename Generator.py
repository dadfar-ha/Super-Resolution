!python -c 'import torch; print(torch.cuda.is_available())'

vgg = models.vgg19(pretrained=True).to(device)

class ResidualDenseBlock(n.Module):
    def __init__(self,in_channel = 64,inc_channel = 32, beta = 0.2):
        super().__init__()
        self.conv1 = n.Conv2d(in_channel, inc_channel, 3, 1, 1)
        self.conv2 = n.Conv2d(in_channel + inc_channel, inc_channel, 3, 1, 1)
        self.conv3 = n.Conv2d(in_channel + 2 * inc_channel, inc_channel, 3, 1, 1)
        self.conv4 = n.Conv2d(in_channel + 3 * inc_channel, inc_channel, 3, 1, 1)
        self.conv5 = n.Conv2d(in_channel + 4 * inc_channel,  in_channel, 3, 1, 1)
        self.lrelu = n.LeakyReLU(negative_slope=0.2, inplace=True)
        self.b = beta
        
    def forward(self, x):
        block1 = self.lrelu(self.conv1(x))
        block2 = self.lrelu(self.conv2(torch.cat((block1, x), dim = 1)))
        block3 = self.lrelu(self.conv3(torch.cat((block2, block1, x), dim = 1)))
        block4 = self.lrelu(self.conv4(torch.cat((block3, block2, block1, x), dim = 1)))
        out = self.conv5(torch.cat((block4, block3, block2, block1, x), dim = 1))
        
        return x + self.b * out

class ResidualInResidualDenseBlock(n.Module):
    def __init__(self, in_channel = 64, out_channel = 32, beta = 0.2):
        super().__init__()
        self.RDB = ResidualDenseBlock(in_channel, out_channel)
        self.b = beta
    
    def forward(self, x):
        out = self.RDB(x)
        out = self.RDB(out)
        out = self.RDB(out)
        
        return x + self.b * out

class Generator(n.Module):
    def __init__(self,in_channel = 3, out_channel = 3, noRRDBBlock = 23):
        super().__init__()   
        self.conv1 = n.Conv2d(3, 64, 3, 1, 1)

        RRDB = ResidualInResidualDenseBlock()
        RRDB_layer = []
        for i in range(noRRDBBlock):
            RRDB_layer.append(RRDB)
        self.RRDB_block =  n.Sequential(*RRDB_layer)

        self.RRDB_conv2 = n.Conv2d(64, 64, 3, 1, 1)
        self.upconv = n.Conv2d(64, 64, 3, 1, 1)

        self.out_conv = n.Conv2d(64, 3, 3, 1, 1)
    
    def forward(self, x):
        first_conv = self.conv1(x)
        RRDB_full_block = torch.add(self.RRDB_conv2(self.RRDB_block(first_conv)),first_conv)
        upconv_block1 = self.upconv(f.interpolate(RRDB_full_block, scale_factor = 2))
        out = self.upconv(f.interpolate(upconv_block1, scale_factor = 2))
        out = self.out_conv(out)
        
        return out

gen = Generator().to(device)