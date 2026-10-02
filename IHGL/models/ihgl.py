import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.sparse as sp
import numpy as np

from IHGL.data.preprocessing import load_adjacency


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def compute_support_gwn(adj, device=None):
    adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
    support = [torch.tensor(i).to(device) for i in adj_mx]
    return support


def get_cross_feature_adj(adj_in, num_node, feature):
    adj_out = np.zeros((num_node * feature, num_node * feature))
    for i in range(adj_in.shape[0]):
        for j in range(adj_in.shape[1]):
            weight = adj_in[i, j]

            x = i * feature
            y = j * feature
            for f in range(feature):
                adj_out[x + f, y + f] = weight
                adj_out[y + f, x + f] = weight
    return adj_out


class IHGLModel_OOD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.configs = config
        self.model = IHGLModel(config)
        self.missing_pattern = config.missing_pattern
        if config.missing_pattern == "mnar":
            # A leading singleton dimension lets the final, smaller batch use the
            # same learnable environment embedding as full batches.
            self.missing_emb = nn.Parameter(
                torch.randn(1, config.seq_len, config.num_nodes, config.feature)
            )
        miss_projetion_1 = []
        miss_projetion_2 = []
        for i in range(config.missing_env_num):
            miss_projetion_1.append(
                nn.Conv1d(
                    in_channels=config.feature,
                    out_channels=config.channels,
                    kernel_size=1,
                    bias=True,
                )
            )
            miss_projetion_2.append(
                nn.Conv1d(
                    in_channels=config.channels,
                    out_channels=config.feature,
                    kernel_size=1,
                    bias=True,
                )
            )
        self.miss_proj_1 = nn.ModuleList(miss_projetion_1)
        self.miss_proj_2 = nn.ModuleList(miss_projetion_2)
        self.metirc = nn.MSELoss()

    def forward(self, x_obs, observed_tp, cond_mask):
        x = x_obs * cond_mask
        B, T, N, F = x.shape
        res = []
        loss = []
        if self.missing_pattern == "mcar":
            random_array = torch.rand(B, T, N, F).to(self.configs.device)
            for i in range(self.configs.missing_env_num):
                self.missing_array = self.miss_proj_1[i](
                    random_array.permute(0, 3, 1, 2).flatten(2)
                )
                self.missing_array = (
                    self.miss_proj_2[i](self.missing_array).unflatten(2, (T, N)).permute(0, 2, 3, 1)
                )
                training_mask = torch.ones_like(self.missing_array).to(self.configs.device)
                training_mask[self.missing_array <= self.configs.mask_rate] = 0
                training_mask[self.missing_array > self.configs.mask_rate] = 1
                training_mask[torch.where(cond_mask == 0)] = 0
                predict_data = self.model(x, observed_tp, training_mask)
                res.append(predict_data)
                loss.append(self.metirc(predict_data * cond_mask, x_obs * cond_mask))
        elif self.missing_pattern == "mnar":
            for i in range(self.configs.missing_env_num):
                missing_emb = self.missing_emb.expand(B, -1, -1, -1)
                self.missing_array = self.miss_proj_1[i](missing_emb.permute(0, 3, 1, 2).flatten(2))
                self.missing_array = (
                    self.miss_proj_2[i](self.missing_array).unflatten(2, (T, N)).permute(0, 2, 3, 1)
                )
                training_mask = torch.ones_like(self.missing_array).to(self.configs.device)
                training_mask[self.missing_array <= self.configs.mask_rate] = 0
                training_mask[self.missing_array > self.configs.mask_rate] = 1
                training_mask[torch.where(cond_mask == 0)] = 0
                predict_data = self.model(x, observed_tp, training_mask)
                res.append(predict_data)
                loss.append(self.metirc(predict_data * cond_mask, x_obs * cond_mask))
        elif self.missing_pattern == "mar":
            for i in range(self.configs.missing_env_num):
                self.missing_array = self.miss_proj_1[i](x.permute(0, 3, 1, 2).flatten(2))
                self.missing_array = (
                    self.miss_proj_2[i](self.missing_array).unflatten(2, (T, N)).permute(0, 2, 3, 1)
                )
                training_mask = torch.ones_like(self.missing_array).to(self.configs.device)
                training_mask[self.missing_array <= self.configs.mask_rate] = 0
                training_mask[self.missing_array > self.configs.mask_rate] = 1
                training_mask[torch.where(cond_mask == 0)] = 0
                predict_data = self.model(x, observed_tp, training_mask)
                res.append(predict_data)
                loss.append(self.metirc(predict_data * cond_mask, x_obs * cond_mask))
        concat_loss = torch.stack(loss).flatten()
        var, mean = torch.var_mean(concat_loss, unbiased=False)
        penalty = torch.sum(var) * self.configs.var_loss_weight
        if self.configs.with_var_loss:
            mean += penalty
        res_mean = torch.mean(torch.stack(res), dim=0)
        return res_mean, mean


class IHGLModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.channels = configs.channels
        self.emb_time_dim = configs.timeemb
        self.emb_feature_dim = configs.featureemb
        self.target_dim = configs.feature * configs.num_nodes
        self.device = torch.device(configs.device)
        side_dim = self.emb_time_dim + self.emb_feature_dim + 1
        node_num = configs.num_nodes
        feature_num = configs.feature

        self.adj = load_adjacency(configs.data_root, configs.dataset)
        if self.adj.shape != (node_num, node_num):
            raise ValueError(
                f"Adjacency shape {self.adj.shape} does not match num_nodes={node_num}"
            )
        self.support = compute_support_gwn(self.adj, device=self.device)
        self.adj_cross_feature = get_cross_feature_adj(self.adj, node_num, feature_num)
        self.support_cross_feature = compute_support_gwn(self.adj_cross_feature, device=self.device)

        # ParameterList keeps one adaptive graph per feature registered in the
        # state dict. The old implementation overwrote the same attribute.
        self.nodevec_node_1 = nn.ParameterList(
            [
                nn.Parameter(torch.randn(node_num, 10, device=self.device))
                for _ in range(feature_num)
            ]
        )
        self.nodevec_node_2 = nn.ParameterList(
            [
                nn.Parameter(torch.randn(10, node_num, device=self.device))
                for _ in range(feature_num)
            ]
        )
        for nodevec_1, nodevec_2 in zip(self.nodevec_node_1, self.nodevec_node_2):
            self.support.append([nodevec_1, nodevec_2])

        self.nodevec_feature_1 = nn.Parameter(torch.randn(feature_num, 10, device=self.device))
        self.nodevec_feature_2 = nn.Parameter(torch.randn(10, feature_num, device=self.device))
        self.support_feature = (self.nodevec_feature_1, self.nodevec_feature_2)

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        self.input_projection = Conv1d_with_init(1, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    nheads=configs.nheads,
                    node_num=configs.num_nodes,
                    feature_num=configs.feature,
                    device=self.device,
                )
                for _ in range(configs.layers)
            ]
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, inputdim, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, inputdim * K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K*inputdim,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,inputdim*K,L)

        side_mask = cond_mask.unsqueeze(1).reshape(B, 1, inputdim * K, L)  # (B,1,inputdim, K, L)
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, x, observed_tp, cond_mask):
        # x: B, L, K(node), inputdim(channel)
        cond_mask = cond_mask[: x.shape[0], :, :, :]
        x = x * cond_mask

        x = x.permute(0, 3, 2, 1)
        cond_mask = cond_mask.permute(0, 3, 2, 1)  # B, inputdim, K, L

        B, inputdim, K, L = x.shape

        cond_info = self.get_side_info(observed_tp, cond_mask)

        x = x.unsqueeze(1)
        x = x.reshape(B, 1, inputdim * K * L)
        x = self.input_projection(x)  # B, channel, inputdim * K * L
        x = F.relu(x)
        x = x.reshape(B, self.channels, inputdim, K, L)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(
                x, cond_info, self.support, self.support_feature, self.support_cross_feature
            )
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, inputdim * K * L)
        x = self.output_projection1(x)  # (B,channel,inputdim*K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,inputdim*K*L)
        x = x.unsqueeze(1).reshape(B, inputdim, K, L)
        x = x.permute(0, 3, 2, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, nheads, node_num, feature_num, device):
        super().__init__()
        self.Node_Scale_GCN = Adaptive_Node_Scale_GCN(channels, node_num, feature_num, device)
        self.Feature_Scale_GCN = Adaptive_Feature_Scale_GCN(channels, node_num, feature_num)
        # self.Cross_Feature_GCN = Adaptive_Cross_Feature_GCN(channels, node_num, feature_num)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.node_layer = get_torch_trans(heads=nheads, layers=1, channels=channels * feature_num)

        self.gcn_projection = Conv1d_with_init(3 * channels, channels, 1)
        # self.node_projection = Conv1d_with_init(channels*feature_num, channels*feature_num, 1)

    def forward_time(self, y, base_shape):
        B, channel, input_dim, K, L = base_shape
        if L == 1:
            return y
        y = (
            y.reshape(B, channel, input_dim, K, L)
            .permute(0, 2, 3, 1, 4)
            .reshape(B * input_dim * K, channel, L)
        )
        # permute: L, B * input_dim * K, channel
        # out: B * input_dim * K, channel, L
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        # B, channel, input_dim, K, L
        y = (
            y.reshape(B, input_dim, K, channel, L)
            .permute(0, 3, 1, 2, 4)
            .reshape(B, channel, input_dim * K * L)
        )
        return y

    def forward_feature(self, y, base_shape, support, support_feature, support_cross_feature):
        B, channel, input_dim, K, L = base_shape
        if K == 1:
            return y

        y_local_node = self.Node_Scale_GCN(y, base_shape, support)
        y_local_feature = self.Feature_Scale_GCN(y, base_shape, support_feature)
        # y_local_cross_feature = self.Cross_Feature_GCN(y, base_shape, support_cross_feature)

        y = torch.stack([y, y_local_node, y_local_feature], dim=1).reshape(B, 3 * channel, -1)
        y = self.gcn_projection(y)

        y = (
            y.reshape(B, channel, input_dim, K, L)
            .permute(0, 4, 1, 2, 3)
            .reshape(B * L, channel, input_dim * K)
        )
        # permute: input_dim*K, B * L, channel
        # out: B * L, channel, input_dim*K
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = (
            y.reshape(B, L, channel, input_dim * K)
            .permute(0, 2, 3, 1)
            .reshape(B, channel, input_dim * K * L)
        )
        return y

    def forward(self, x, cond_info, support, support_feature, support_cross_feature):
        B, channel, input_dim, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, input_dim * K * L)

        y = self.forward_time(x, base_shape)
        y = self.forward_feature(
            y, base_shape, support, support_feature, support_cross_feature
        )  # (B,channel,input_dim*K*L)
        y = self.mid_projection(y)  # (B,2*channel,input_dim*K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, input_dim * K * L)

        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


def tempsigmoid(x, temperature=1.5):
    return torch.sigmoid(x / temperature)


def tempsoftmax(x, temperature=1.5):
    return torch.nn.functional.softmax(x / temperature, dim=1)


class Adaptive_Node_Scale_GCN(nn.Module):
    def __init__(
        self, channels, node_num, feature_num, device, order=2, include_self=True, is_adp=True
    ):
        super().__init__()
        self.order = order
        self.include_self = include_self
        self.feature_num = feature_num
        c_in = channels
        c_out = channels
        self.support_len = 2
        self.is_adp = is_adp
        if is_adp:
            self.support_len += 1
        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, kernel_size=1).to(device) for _ in range(feature_num)]
        )
        # self.node_encoder = Conv1d_with_init(feature_num*channels, feature_num*channels, 1)
        self.importance = nn.Linear(node_num, node_num)

    def forward(self, x, base_shape, support_adp):
        B, channel, input_dim, K, L = base_shape
        if K == 1:
            return x
        if self.is_adp:
            nodevec_node1 = []
            nodevec_node2 = []
            support_feautre = []
            for iter in range(self.feature_num):
                nodevec_node1 = support_adp[2 + iter][0]
                nodevec_node2 = support_adp[2 + iter][1]
                importance = self.importance(nodevec_node1.T).T
                nodevec_node1 = nodevec_node1 * importance
                adp = F.softmax(F.relu(torch.mm(nodevec_node1, nodevec_node2)), dim=1)
                support_feautre = support_feautre + [adp]
            support = support_adp[0:2]
        else:
            support = support_adp
        x = (
            x.reshape(B, channel, input_dim, K, L)
            .permute(2, 0, 4, 1, 3)
            .reshape(input_dim, B * L, channel, K)
        )
        # x = self.node_encoder(x)
        x = torch.unsqueeze(x, -1)  # input_dim, B * L, channel, K, 1

        res = []
        for iter in range(self.feature_num):
            x_feature_i = x[iter]  # B * L, channel, K, 1
            out = [x_feature_i] if self.include_self else []
            if type(support) is not list:
                support = [support]
            support_feature_i = support + [support_feautre[iter]]
            for a in support_feature_i:
                x1 = torch.einsum("ncvl,wv->ncwl", (x_feature_i, a)).contiguous()
                out.append(x1)
                for k in range(2, self.order + 1):
                    x2 = torch.einsum("ncvl,wv->ncwl", (x1, a)).contiguous()
                    out.append(x2)
                    x1 = x2
            out = torch.cat(out, dim=1)
            out = self.mlp[iter](out)
            out = out.squeeze(-1)
            out = out.reshape(B, L, channel, K)
            res.append(out)
        res = torch.stack(res, dim=2).permute(0, 3, 2, 4, 1).reshape(B, channel, input_dim * K * L)
        return res
        # out = out.reshape(B, L, input_dim, channel, K).permute(0, 3, 2, 4, 1).reshape(B, channel, input_dim * K * L)
        # return out


class Adaptive_Feature_Scale_GCN(nn.Module):
    def __init__(self, channels, node_num, feature_num, order=2, include_self=True, is_adp=True):
        super().__init__()
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels
        self.support_len = 0
        self.is_adp = is_adp
        if is_adp:
            self.support_len += 1
        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.importance = nn.Linear(feature_num, feature_num)

    def forward(self, x, base_shape, support_feature):
        B, channel, input_dim, K, L = base_shape
        if K == 1:
            return x
        if self.is_adp:
            nodevec_feature1 = support_feature[0]
            nodevec_feature2 = support_feature[1]
            support = []
        x = (
            x.reshape(B, channel, input_dim, K, L)
            .permute(0, 4, 3, 1, 2)
            .reshape(B * L * K, channel, input_dim)
        )
        x = torch.unsqueeze(x, -1)  # B * L, channel*input_dim, K, 1

        out = [x] if self.include_self else []
        if self.is_adp:
            importance = self.importance(nodevec_feature1.T).T
            nodevec_feature1 = nodevec_feature1 * importance
            adp = F.softmax(
                F.relu(torch.mm(nodevec_feature1, nodevec_feature2)), dim=1
            )  # input_dim * input_dim
            support += [adp]
        for a in support:
            x1 = torch.einsum("ncvl,wv->ncwl", (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum("ncvl,wv->ncwl", (x1, a)).contiguous()
                out.append(x2)
                x1 = x2

        out = torch.cat(out, dim=1)

        out = self.mlp(out)
        out = out.squeeze(-1)
        out = (
            out.reshape(B, L, K, channel, input_dim)
            .permute(0, 3, 4, 2, 1)
            .reshape(B, channel, input_dim * K * L)
        )
        return out


class Adaptive_Cross_Feature_GCN(nn.Module):
    def __init__(self, channels, node_num, feature_num, order=2, include_self=True, is_adp=True):
        super().__init__()
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels
        self.support_len = 2
        self.is_adp = is_adp
        if is_adp:
            self.support_len += 1
        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.importance = nn.Linear(node_num * feature_num, node_num * feature_num)

    def forward(self, x, base_shape, support_adp):
        B, channel, input_dim, K, L = base_shape
        if K == 1:
            return x
        if self.is_adp:
            nodevec_cross_feature_1 = support_adp[-1][0]
            nodevec_cross_feature_2 = support_adp[-1][1]
            support = support_adp[:-1]
        else:
            support = support_adp
        x = (
            x.reshape(B, channel, input_dim, K, L)
            .permute(0, 4, 1, 2, 3)
            .reshape(B * L, channel, input_dim * K)
        )
        x = torch.unsqueeze(x, -1)  # B * L, channel*input_dim, K, 1

        out = [x] if self.include_self else []
        if type(support) is not list:
            support = [support]
        if self.is_adp:
            importance = self.importance(nodevec_cross_feature_1.T).T
            nodevec_cross_feature_1 = nodevec_cross_feature_1 * importance
            adp = F.softmax(
                F.relu(torch.mm(nodevec_cross_feature_1, nodevec_cross_feature_2)), dim=1
            )
            support = support + [adp]
        for a in support:
            x1 = torch.einsum("ncvl,wv->ncwl", (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum("ncvl,wv->ncwl", (x1, a)).contiguous()
                out.append(x2)
                x1 = x2

        out = torch.cat(out, dim=1)

        out = self.mlp(out)
        out = out.squeeze(-1)
        out = (
            out.reshape(B, L, channel, input_dim, K)
            .permute(0, 2, 3, 4, 1)
            .reshape(B, channel, input_dim * K * L)
        )
        return out
