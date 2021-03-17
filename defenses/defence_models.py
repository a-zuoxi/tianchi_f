import torch
# model = models.resnet50(pretrained=False)
weight_wsdan = '../defenses/weights/model_renamed/inception_v4.ckpt'
checkpoint = torch.load(weight_wsdan)
print(checkpoint)
# sta_dic = checkpoint['state_dict']
# # model.load_state_dict(sta_dic)
# pretext_model = torch.load(weight_wsdan)['state_dict']
# model_dict = model.state_dict()
# model_new_dict = {}
# for k, v in pretext_model.items():
#     k = k.replace("module.", "")
#     model_new_dict[k] = v
# state_dict = {k: v for k, v in model_new_dict.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# model.load_state_dict(model_dict)
