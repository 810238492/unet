from calendar import c
from doctest import FAIL_FAST
from re import A
from turtle import forward
from pip import main
from torch import  nn, relu
from torch.nn import functional as F
import torch

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)



class UNet(nn.Module):

    
    def __init__(self, in_channel=3,n=4,first_out_channel=64,out_channel=3):
        super(UNet,self).__init__()
        self.con,self.down,self.up=[],[],[]
        for i in range(n):
            print(in_channel)
            print(first_out_channel)
            self.con.append(Conv_Block(in_channel,first_out_channel))
            self.down.append(DownSample(first_out_channel))
            in_channel=first_out_channel
            first_out_channel*=2
        
        
        
        
        for i in range(n):
            self.con.append(Conv_Block(in_channel,first_out_channel))
            self.up.append(UpSample(first_out_channel))
            in_channel=first_out_channel
            first_out_channel//=2

        self.con.append(Conv_Block(in_channel,first_out_channel))
        self.out =nn.Conv2d(first_out_channel,out_channel,3,1,1)
        self.Th=nn.Sigmoid()
        # self.c1=Conv_Block(3,64)
        # self.d1=DownSample(64)
        # self.c2=Conv_Block(64,128)
        # self.d2=DownSample(128)
        # self.c3=Conv_Block(128,256)
        # self.d3=DownSample(256)
        # self.c4=Conv_Block(256,512)
        # self.d4=DownSample(512)


        # self.c5=Conv_Block(512,1024)


        # self.u1=UpSample(1024)
        # self.c6=Conv_Block(1024,512)
        # self.u2=UpSample(512)
        # self.c7=Conv_Block(512,256)
        # self.u3=UpSample(256)
        # self.c8=Conv_Block(256,128)
        # self.u4=UpSample(128)

        # self.c9=Conv_Block(128,64)
        # self.out =nn.Conv2d(64,3,3,1,1)
        # self.Th=nn.Sigmoid()


    def forward(self,x,n=4):

        R=[]
        A=[]
        R.append(self.con[0](x))
        for i in range(n):
            R.append(self.con[i+1](self.down[i](R[i])))

        A.append(self.con[n+1](self.up[0](R[n],R[n-1])))
        
        for i in range(n-1):
            A.append(self.con[i+n+2](self.up[i+1](A[i],R[n-i-2])))
        # R1=self.c1(x)
        # R2 = self.c2(self.d1(R1))
        # R3 = self.c3(self.d2(R2))
        # R4 = self.c4(self.d3(R3))
        # R5 = self.c5(self.d4(R4))
        # O1 = self.c6(self.u1(R5,R4))
        # O2 = self.c7(self.u2(O1,R3))
        # O3 = self.c8(self.u3(O2,R2))
        # O4 = self.c9(self.u4(O3,R1))
        
        return self.Th(self.out(A[n-1]))

if __name__ =='__main__':
    x=torch.randn(2,6,512,1024)
    net=UNet(6,4,64,3)
    print(net(x).shape)
