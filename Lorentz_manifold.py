import torch

thre = {torch.float64: 1e-12}

def arcosh(data_point):
    return torch.log(torch.clamp(data_point, min=1+thre[data_point.dtype]) + torch.sqrt(data_point*data_point - 1))


def cosh(data_point):
    return torch.cosh(data_point.data.clamp_(max=1000))


def sinh(data_point):
    return torch.sinh(data_point.data.clamp_(max=700))


class Lorentz_m:
    def __init__(self):
        self.curvature = -1


    def lorentz_inner_product_v(self, vector_a, vector_b):
        mul_ab = vector_a * vector_b
        return torch.sum(mul_ab) - 2 * mul_ab[..., 0]


    def lorentz_inner_product_f(self, vector_a, vector_b):
        mul_ab = vector_a * vector_b
        return torch.sum(mul_ab, dim=-1, keepdim=True) - 2 * mul_ab[..., 0:1]


    def Exp_Map(self, x, v):
        v_lnorm = (self.lorentz_inner_product_f(v, v)).clamp(min=thre[x.dtype]).sqrt()
        return x * cosh(v_lnorm) + sinh(v_lnorm) * v / v_lnorm


    def Log_Map(self, x, y):
        lambda_xy = - self.lorentz_inner_product_f(x, y)
        return (arcosh(lambda_xy) / (lambda_xy.pow(2) - 1).sqrt()) * (y - lambda_xy * x)


    def PT(self, x, y, v):
        lambda_xy =  self.lorentz_inner_product_f(x, y).expand_as(v)
        lambda_vy =  self.lorentz_inner_product_f(v, y).expand_as(v)
        return v + lambda_vy / (1 - lambda_xy) * (x + y)
