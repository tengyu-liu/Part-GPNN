"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn
import torch.autograd


class MessageFunction(torch.nn.Module):
    def __init__(self, message_def, args):
        super(MessageFunction, self).__init__()
        self.m_definition = ''
        self.m_function = None
        self.args = {}
        self.learn_args = torch.nn.ParameterList([])
        self.learn_modules = torch.nn.ModuleList([])
        self.__set_message(message_def, args)

    # Message from h_v to h_w through e_vw
    def forward(self, h_v, h_w, e_vw, args=None):
        if self.m_definition == 'linear':
            message = torch.autograd.Variable(torch.zeros(e_vw.size()[0], self.args['message_size'], e_vw.size()[2]))
            if hasattr(args, 'cuda') and args.cuda:
                message = message.cuda()

            for i_node in range(e_vw.size()[2]):
                message[:, :, i_node] = self.learn_modules[0](e_vw[:, :, i_node]) + self.learn_modules[1](h_w[:, :, i_node])
            return message

        if self.m_definition == 'linear_edge':
            message = torch.autograd.Variable(torch.zeros(e_vw.size()[0], self.args['message_size'], e_vw.size()[2]))
            if hasattr(args, 'cuda') and args.cuda:
                message = message.cuda()

            for i_node in range(e_vw.size()[2]):
                message[:, :, i_node] = self.learn_modules[0](e_vw[:, :, i_node])
            return message

        if self.m_definition == 'linear_concat':
            message = torch.autograd.Variable(torch.zeros(e_vw.size()[0], self.args['message_size'], e_vw.size()[2]))
            if hasattr(args, 'cuda') and args.cuda:
                message = message.cuda()

            for i_node in range(e_vw.size()[2]):
                message[:, :, i_node] = torch.cat([self.learn_modules[0](e_vw[:, :, i_node]), self.learn_modules[1](h_w[:, :, i_node])], 1)
            return message

        if self.m_definition == 'linear_concat_relu':
            message = torch.autograd.Variable(torch.zeros(e_vw.size()[0], self.args['message_size'], e_vw.size()[2]))
            if hasattr(args, 'cuda') and args.cuda:
                message = message.cuda()

            # Optimized
            e_vw_shape = e_vw.shape
            h_w_shape = h_w.shape
            e_vw = e_vw.permute([0,2,1]).contiguous().view([-1, e_vw_shape[1]])
            h_w = h_w.permute([0,2,1]).contiguous().view([-1, h_w_shape[1]])
            e_vw = self.learn_modules[0](e_vw).contiguous().view([e_vw_shape[0], e_vw_shape[2], -1]).permute([0,2,1])
            h_w = self.learn_modules[1](h_w).contiguous().view([h_w_shape[0], h_w_shape[2], -1]).permute([0,2,1])
            message = torch.cat([e_vw, h_w], dim=1)

            # for i_node in range(e_vw.size()[2]):
            #     message[:, :, i_node] = torch.cat([self.learn_modules[0](e_vw[:, :, i_node]), self.learn_modules[1](h_w[:, :, i_node])], 1)
            return message

        print('WARNING!: Message Function has not been set correctly\n\tIncorrect definition ' + message_def)
        quit()

    # Set a message function
    def __set_message(self, message_def, args):
        self.m_definition = message_def.lower()
        self.args = args

        # self.m_function = {
        #     'linear':           self.m_linear,
        #     'linear_edge':      self.m_linear_edge,
        #     'linear_concat':    self.m_linear_concat,
        #     'linear_concat_relu':    self.m_linear_concat_relu,
        # }.get(self.m_definition, None)

        # if self.m_function is None:
        #     print('WARNING!: Message Function has not been set correctly\n\tIncorrect definition ' + message_def)
        #     quit()

        init_parameters = {
            'linear':           self.init_linear,
            'linear_edge':      self.init_linear_edge,
            'linear_concat':    self.init_linear_concat,
            'linear_concat_relu':    self.init_linear_concat_relu,
        }.get(self.m_definition, lambda x: (torch.nn.ParameterList([]), torch.nn.ModuleList([]), {}))

        init_parameters()

    # Get the name of the used message function
    def get_definition(self):
        return self.m_definition

    # Get the message function arguments
    def get_args(self):
        return self.args

    # Definition of message functions
    # Combination of linear transformation of edge features and node features
    def m_linear(self, h_v, h_w, e_vw, args):
        message = torch.autograd.Variable(torch.zeros(e_vw.size()[0], self.args['message_size'], e_vw.size()[2]))
        if hasattr(args, 'cuda') and args.cuda:
            message = message.cuda()

        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = self.learn_modules[0](e_vw[:, :, i_node]) + self.learn_modules[1](h_w[:, :, i_node])
        return message

    def init_linear(self):
        edge_feature_size = self.args['edge_feature_size']
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['message_size']
        self.learn_modules.append(torch.nn.Linear(edge_feature_size, message_size, bias=True))
        self.learn_modules.append(torch.nn.Linear(node_feature_size, message_size, bias=True))

    # Linear transformation of edge features
    def m_linear_edge(self, h_v, h_w, e_vw, args):
        message = torch.autograd.Variable(torch.zeros(e_vw.size()[0], self.args['message_size'], e_vw.size()[2]))
        if hasattr(args, 'cuda') and args.cuda:
            message = message.cuda()

        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = self.learn_modules[0](e_vw[:, :, i_node])
        return message

    def init_linear_edge(self):
        edge_feature_size = self.args['edge_feature_size']
        message_size = self.args['message_size']
        self.learn_modules.append(torch.nn.Linear(edge_feature_size, message_size, bias=True))

    # Concatenation of linear transformation of edge features and node features
    def m_linear_concat(self, h_v, h_w, e_vw, args):
        message = torch.autograd.Variable(torch.zeros(e_vw.size()[0], self.args['message_size'], e_vw.size()[2]))
        if hasattr(args, 'cuda') and args.cuda:
            message = message.cuda()

        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = torch.cat([self.learn_modules[0](e_vw[:, :, i_node]), self.learn_modules[1](h_w[:, :, i_node])], 1)
        return message

    def init_linear_concat(self):
        edge_feature_size = self.args['edge_feature_size']
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['message_size']/2
        self.learn_modules.append(torch.nn.Linear(edge_feature_size, message_size, bias=True))
        self.learn_modules.append(torch.nn.Linear(node_feature_size, message_size, bias=True))

    # Concatenation of linear transformation of edge features and node features with ReLU
    def m_linear_concat_relu(self, h_v, h_w, e_vw, args):
        message = torch.autograd.Variable(torch.zeros(e_vw.size()[0], self.args['message_size'], e_vw.size()[2]))
        if hasattr(args, 'cuda') and args.cuda:
            message = message.cuda()

        # Optimized
        e_vw_shape = e_vw.shape
        h_w_shape = h_w.shape
        e_vw = e_vw.permute([0,2,1]).contiguous().view([-1, e_vw_shape[1]])
        h_w = h_w.permute([0,2,1]).contiguous().view([-1, h_w_shape[1]])
        e_vw = self.learn_modules[0](e_vw).contiguous().view([e_vw_shape[0], e_vw_shape[2], -1]).permute([0,2,1])
        h_w = self.learn_modules[1](h_w).contiguous().view([h_w_shape[0], h_w_shape[2], -1]).permute([0,2,1])
        message = torch.cat([e_vw, h_w], dim=1)

        # for i_node in range(e_vw.size()[2]):
        #     message[:, :, i_node] = torch.cat([self.learn_modules[0](e_vw[:, :, i_node]), self.learn_modules[1](h_w[:, :, i_node])], 1)
        return message

    def init_linear_concat_relu(self):
        edge_feature_size = self.args['edge_feature_size']
        node_feature_size = self.args['node_feature_size']
        message_size = int(self.args['message_size']/2)
        self.learn_modules.append(torch.nn.Linear(edge_feature_size, message_size, bias=True))
        self.learn_modules.append(torch.nn.Linear(node_feature_size, message_size, bias=True))


def main():
    pass


if __name__ == '__main__':
    main()
