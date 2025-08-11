import torch
import torchvision.transforms as transforms
from torchvision.models import (
    resnet152, ResNet152_Weights,
    densenet121, DenseNet121_Weights,
    vgg16, VGG16_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    inception_v3, Inception_V3_Weights
)

class LCA:
    def __init__(self, feature_size, dictionary_num, UPDATE_DICT, dictionary_iter, 
                 neuron_iter, lr_dictionary, lr_neuron, landa):
        self.feature_size = feature_size
        self.dict_num = dictionary_num
        self.UPDATE_DICT = UPDATE_DICT
        self.dict_iter = dictionary_iter
        self.neuron_iter = neuron_iter
        self.lr_dict = lr_dictionary
        self.lr_neuron = lr_neuron
        self.landa = landa
        self.dictionary = None
        self.data = None
        self.input = None
        self.a = None
        self.u = None

    def lca_update(self, n, phi, G, device, fn_train, fn_test):
        u_list = [torch.zeros([1, self.dict_num]).to(device)]
        a_list = [self.threshold(u_list[0], 'soft', True, self.landa).to(device)]

        if n == 0:
            dict = self.dictionary.reshape(self.dict_num, -1)
            phi = dict.T
            phi = phi.to(device)
            I = torch.eye(self.dict_num).to(device)
            G = torch.mm(phi.T, phi) - I
            G = G.to(device)
            input = self.input.detach().reshape(-1)
        elif n == 1:
            input = self.input.reshape(fn_train, 3 * self.feature_size * self.feature_size)
        elif n == 2:
            input = self.input.reshape(fn_test, 3 * self.feature_size * self.feature_size)
        else:
            input = self.input.reshape(50, -1)

        S = input.T
        b = torch.matmul(S.T, phi)
        for t in range(self.neuron_iter):
            u = self.neuron_update(u_list[t], a_list[t], b, G)
            u_list.append(u)
            a = self.threshold(u, 'soft', True, self.landa)
            a_list.append(a)

        self.a = a_list[-1]
        self.u = u_list[-1]

    def loss(self, feature_dim=2048):
        s = self.input.reshape(feature_dim, 1)
        phi = self.dictionary.reshape(self.dict_num, -1).T
        a = self.a
        residual = s - torch.mm(phi, a.T)
        approximation_loss = .5 * torch.linalg.norm(residual, 'fro')
        sparsity_loss = self.landa * torch.sum(torch.abs(a))
        loss = approximation_loss + sparsity_loss
        print('Loss: {:.2f}'.format(loss.item()), 'approximation loss: {:.2f}'.format(approximation_loss.item()), 'sparsity loss: {:.2f}'.format(sparsity_loss.item()))
        return loss

    def dict_update(self, device):
        phi = self.dictionary.reshape(self.dict_num, -1).T
        phi = phi.to(device)        
        S = self.input.reshape(-1, 1)
        d_phi = torch.matmul((S.reshape(-1, 50)-torch.matmul(phi, self.a.T)), self.a)
        d_dict = d_phi.T.reshape([self.dict_num, 1000])
        d_dict = d_dict.cpu()
        self.dictionary = self.dictionary + d_dict * self.lr_dict

    def threshold(self, u, type, rectify, landa):
        u_zeros = torch.zeros_like(u)
        if type == 'soft':
            if rectify:
                a_out = torch.where(torch.greater(u, landa), u - landa, u_zeros)
            else:
                a_out = torch.where(torch.ge(u, landa), u - landa,
                                    torch.where(torch.le(u, - landa), u + landa, u_zeros))
        elif type == 'hard':
            if rectify:
                a_out = torch.where(torch.gt(u, landa), u, u_zeros)
            else:
                a_out = torch.where(torch.ge(u, landa), u,
                                    torch.where(torch.le(u, -landa), u, u_zeros))
        else:
            assert False, (f'Parameter thresh_type must be "soft" or "hard", not {type}')
        return a_out

    def neuron_update(self, u_in, a_in, b, G):
        du = b - torch.mm(a_in, G) - u_in
        u_out = u_in + self.lr_neuron * du
        return u_out

def normalize(M):
    sigma = torch.sum(M * M)
    return M / torch.sqrt(sigma)

def get_model_and_transforms(model_name):
    models = {
        'resnet152': (resnet152, ResNet152_Weights.IMAGENET1K_V2),
        'densenet121': (densenet121, DenseNet121_Weights.IMAGENET1K_V1),
        'vgg16': (vgg16, VGG16_Weights.IMAGENET1K_V1),
        'efficientnet': (efficientnet_v2_s, EfficientNet_V2_S_Weights.IMAGENET1K_V1),
        'mobilenet': (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        'inception': (inception_v3, Inception_V3_Weights.IMAGENET1K_V1)
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Available: {list(models.keys())}")
    
    model_class, weights = models[model_name]
    model = model_class(weights=weights)
    
    if model_name == 'inception':
        model = torch.nn.Sequential(*list(model.children())[:-2])
    else:
        model = torch.nn.Sequential(*list(model.children())[:-2])
    
    return model, weights.transforms() 