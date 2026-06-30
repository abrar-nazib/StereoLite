import torch                                                                                                                         
from model.designs.StereoLite_yolo_ctx.model import StereoLiteYoloCtx, StereoLiteYoloCtxConfig                                                                                                                                                                                                

m = StereoLiteYoloCtx(StereoLiteYoloCtxConfig(backbone='yolo26n')).cuda().eval()
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'trainable params: {n/1e6:.3f} M')

left  = torch.zeros(2, 3, 256, 512, device='cuda')
right = torch.zeros(2, 3, 256, 512, device='cuda')

with torch.no_grad():
    d = m(left, right)
    d_aux = m(left, right, aux=True)

print('d_final', d.shape)
for k, v in d_aux.items():
    print(f'  {k}: {v.shape}')
