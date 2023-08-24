import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler
from modules import BasicBlock, Bottleneck


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class HRNet(nn.Module):
    def __init__(self, c=48, inchannels=3, bn_momentum=0.1):
        super(HRNet, self).__init__()

        # Input (stem net)
        self.conv1 = nn.Conv2d(inchannels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(256, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]

        x = self.stage4(x)

        return x


class DSPDH_temporal_future(nn.Module):
    def __init__(self, c=32, joints_num=16, num_deconv=2, num_basic_blocks=2, hidden_dim=np.array([192, 384]),
                 temporal_type='local', deltas=False, num_coords=3, local_embedding_size=16, global_embedding_size=100,
                 controller_size=288, future_window_size=20):
        super(DSPDH_temporal_future, self).__init__()

        # temporal branch params
        self.temporal_type = temporal_type
        self.deltas = deltas
        self.hidden_dim = hidden_dim // 4
        self.num_coords = num_coords
        self.local_embedding_size = local_embedding_size
        self.global_embedding_size = global_embedding_size
        self.controller_size = controller_size
        self.future_window_size = future_window_size

        assert self.temporal_type in ['local', 'global', 'local_global']

        # SPDH params
        self.c = c
        self.joints_num = joints_num
        if self.temporal_type == 'local':
            self.branch_channels = self.c + self.joints_num  # input_channels = 32 + 16 = 48
        if self.temporal_type == 'global':
            self.branch_channels = self.c + 1  # input_channels = 32 + 1 = 33
        if self.temporal_type == 'local_global':
            self.branch_channels = self.c + self.joints_num + 1  # input_channels = 32 + 16 + 1 = 49
        self.num_deconv = num_deconv
        self.num_basic_blocks = num_basic_blocks

        self.backbone = HRNet(c=self.c)

        # temporal branch
        if 'local' in self.temporal_type:
            self.embed_past = nn.Sequential(
                nn.Linear(self.num_coords, int(self.local_embedding_size / 2)),
                nn.ReLU(),
                nn.Linear(int(self.local_embedding_size / 2), self.local_embedding_size)
            )
            self.encoder_local_past = nn.GRU(self.local_embedding_size, self.controller_size, 1, batch_first=True)
        if 'global' in self.temporal_type:
            self.embed_past = nn.Sequential(
                nn.Linear(self.joints_num * 3,
                          int(self.joints_num * 3 + ((self.global_embedding_size - (self.joints_num * 3)) // 2))),
                nn.ReLU(),
                nn.Linear(int(self.joints_num * 3 + ((self.global_embedding_size - (self.joints_num * 3)) // 2)),
                          self.global_embedding_size)
            )
            self.encoder_global_past = nn.GRU(self.global_embedding_size, self.controller_size, 1, batch_first=True)

        # temporal deconv layers
        if self.temporal_type == 'local':
            self.deconv_temp = self._make_deconv_temp(self.joints_num, 2, 1)
        if self.temporal_type == 'global':
            self.deconv_temp = self._make_deconv_temp(1, 2, 1)
        if self.temporal_type == 'local_global':
            self.deconv_temp = self._make_deconv_temp(self.joints_num + 1, 2, 1)

        # SPDH branches
        self.uv_deconv_layers = self._make_deconv_layers(self.branch_channels, self.num_deconv,
                                                         self.num_basic_blocks)
        self.uz_deconv_layers = self._make_deconv_layers(self.branch_channels, self.num_deconv,
                                                         self.num_basic_blocks)

        self.uv_final_layers = self._make_final_layers(self.branch_channels, self.num_deconv)
        self.uz_final_layers = self._make_final_layers(self.branch_channels, self.num_deconv)

        # SPDH braches future prediction
        self.uv_fut_final_layers = nn.Sequential(
            nn.Conv2d(self.branch_channels, self.joints_num * self.future_window_size, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(self.joints_num * self.future_window_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.joints_num * self.future_window_size, self.joints_num * self.future_window_size,
                      kernel_size=3, stride=1, padding=1),
        )
        self.uz_fut_final_layers = nn.Sequential(
            nn.Conv2d(self.branch_channels, self.joints_num * self.future_window_size, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(self.joints_num * self.future_window_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.joints_num * self.future_window_size, self.joints_num * self.future_window_size,
                      kernel_size=3, stride=1, padding=1)
        )

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
        else:
            with torch.no_grad():
                self.pe_w, self.pe_h = w, h
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(1, d_model, length))
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)

            return self.pos_embedding

    def _make_sine_position_embedding(self, d_model, temperature=10000, scale=2*math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2)
        return pos  # [h*w, 1, d_model]

    def _make_final_layers(self, input_channels, num_deconv):
        final_layers = []
        output_channels = self.joints_num
        final_layers.append(nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        ))

        for i in range(num_deconv):
            input_channels = self.c
            output_channels = self.joints_num
            final_layers.append(nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ))

        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, input_channels, num_deconv, num_basic_blocks):
        deconv_layers = []
        for i in range(num_deconv):
            final_output_channels = self.joints_num
            input_channels += final_output_channels
            output_channels = self.c
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(2)

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=0.1),
                nn.ReLU(inplace=True)
            ))
            for _ in range(num_basic_blocks):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _make_deconv_temp(self, input_channels, num_deconv, num_basic_blocks):
        deconv_layers = []
        for i in range(num_deconv):
            output_channels = input_channels
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(2)

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=0.1),
                nn.ReLU(inplace=True)
            ))
            for _ in range(num_basic_blocks):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def temporal_branch(self, x_temp, batch_size):
        if self.deltas:  # displacements
            x_temp = x_temp[:, 1:] - x_temp[:, :-1]
        len_temp = x_temp.shape[1]
        if self.temporal_type == 'local':
            x_temp_local = x_temp.permute(0, 2, 1, 3).reshape(-1, len_temp, 3)
            x_temp_feat_local = self.embed_past(x_temp_local)
            output_past, state_past = self.encoder_local_past(x_temp_feat_local)
            state_past = state_past.reshape(batch_size, self.joints_num, 12, 24)
        elif self.temporal_type == 'global':
            x_temp_feat_global = x_temp.reshape(-1, len_temp, self.joints_num * 3)
            x_temp_feat_global = self.embed_past(x_temp_feat_global)
            output_past, state_past = self.encoder_global_past(x_temp_feat_global)
            state_past = state_past.reshape(batch_size, 1, 12, 24)
        else:
            x_temp_local = x_temp.permute(0, 2, 1, 3).reshape(-1, len_temp, 3)
            x_temp_feat_local = self.embed_past(x_temp_local)
            output_past_local, state_past_local = self.encoder_local_past(x_temp_feat_local)
            x_temp_feat_global = x_temp.reshape(-1, len_temp, self.joints_num * 3)
            x_temp_feat_global = self.embed_past(x_temp_feat_global)
            output_past_global, state_past_global = self.encoder_global_past(x_temp_feat_global)
            state_past = torch.cat((state_past_local, state_past_global), dim=1)
            state_past = state_past.reshape(batch_size, self.joints_num + 1, 12, 24)

        return state_past

    def forward(self, x, x_temp):
        x = self.backbone(x)[0]

        batch_size = x_temp.shape[0]
        x_joint = self.temporal_branch(x_temp, batch_size)

        # deconv temporal features
        for i in range(2):
            x_joint = self.deconv_temp[i](x_joint)

        uv_x = torch.cat((x, x_joint), 1)
        uz_x = uv_x.clone()

        fut_uv_map = self.uv_fut_final_layers(uv_x)
        fut_uz_map = self.uz_fut_final_layers(uz_x)

        uv_y = self.uv_final_layers[0](uv_x)
        uz_y = self.uz_final_layers[0](uz_x)

        for i in range(self.num_deconv):
            uv_x = torch.cat([uv_x, uv_y], 1)
            uz_x = torch.cat([uz_x, uz_y], 1)
            uv_x = self.uv_deconv_layers[i](uv_x)
            uz_x = self.uz_deconv_layers[i](uz_x)
            if i < self.num_deconv - 1:
                uv_y = self.uv_final_layers[i + 1](uv_x)
                uz_y = self.uz_final_layers[i + 1](uz_x)
            else:
                uv_map = self.uv_final_layers[i + 1](uv_x)
                uz_map = self.uz_final_layers[i + 1](uz_x)

        return torch.cat([uv_map, uz_map], 1), torch.cat([fut_uv_map, fut_uz_map], 1)


if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    import time

    model = DSPDH_temporal_future(c=32, joints_num=16, temporal_type='local', deltas=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = model.cuda()
    model.train()

    batch_size = 16
    iterations = 1000
    target_curr = torch.rand(batch_size, 32, 192, 384).float().cuda()
    target_fut = torch.rand(batch_size, 640, 48, 96).float().cuda()
    criterion = nn.MSELoss()
    # scaler = GradScaler()

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_param / 1_000_000)
    # with torch.no_grad():
    time_list = list()
    for _ in tqdm(range(iterations)):
        input = torch.rand(batch_size, 3, 192, 384).float().cuda()
        input_t = torch.rand(batch_size, 10, 16, 3).float().cuda()
        old = time.time()
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        out_curr, out_fut = model(input, input_t)
        loss_curr = criterion(out_curr, target_curr)
        loss = loss_curr + criterion(out_fut, target_fut)
        # scaler.scale(loss).backward()
        loss.backward()
        time_list.append(time.time() - old)
    print(time_list)
    print(f"{np.array(time_list).mean() * 1000:.2f}")
