from diffusers import UNet2DConditionModel

input_dir = "/root/autodl-tmp/instruct-pix2pix-model/checkpoint-10000/"
unet = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
for name, module in unet.named_modules():
    if name.endswith("attn2.to_out.0"):  # out proj
        print(name, module.weight.shape)
