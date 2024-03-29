import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

'''
def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return  s / (i + 1)
'''
def dice_coeff(pred, gt):

    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)
    intersection = (p * g).sum()

    p_sum = torch.sum(p * p)
    g_sum = torch.sum(g * g)
    
    return (2. * intersection + 1) / (p_sum + g_sum + 1) 



def Dice_Loss(pred, gt):

    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)
    intersection = (p * g).sum()

    p_sum = torch.sum(p * p)
    g_sum = torch.sum(g * g)
    
    return 1 - ((2. * intersection + 1) / (p_sum + g_sum + 1) )
