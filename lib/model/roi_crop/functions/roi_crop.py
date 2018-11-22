# functions/add.py
import torch
from torch.autograd import Function
from .._ext import roi_crop
import pdb

class RoICropFunction(Function):
    def forward(self, input1, input2):
        self.input1 = input1.clone()
        self.input2 = input2.clone()
        output = input2.new(input2.size()[0], input1.size()[1], input2.size()[1], input2.size()[2]).zero_()
        assert output.device.type == input1.device.type, "output and input1 must on the same device"
        assert output.device.type == input2.device.type, "output and input2 must on the same device"
        if output.device.type == "cpu":
            roi_crop.BilinearSamplerBHWD_updateOutput(input1, input2, output)
        else:
            roi_crop.BilinearSamplerBHWD_updateOutput_cuda(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input1 = self.input1.new(self.input1.size()).zero_()
        grad_input2 = self.input2.new(self.input2.size()).zero_()
        if grad_input1.device.type == "cpu":
            roi_crop.BilinearSamplerBHWD_updateGradInput(self.input1, self.input2, grad_input1, grad_input2,
                                                         grad_output)
        else:
            roi_crop.BilinearSamplerBHWD_updateGradInput_cuda(self.input1, self.input2, grad_input1, grad_input2, grad_output)
        return grad_input1, grad_input2
