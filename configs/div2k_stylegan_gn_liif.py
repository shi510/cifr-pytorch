train_dataset_gan = dict(
    type='SRFixedDownsampled',
    dataset=dict(
        type='DIV2K',
        root='./sr_dataset/DIV2K_train_HR',
        n_repeat=20
    ),
    inp_size=64,
    scale=4
)

train_dataset = dict(
    type='SRImplicitDownsampled',
    dataset=dict(
        type='DIV2K',
        root='./sr_dataset/DIV2K_train_HR',
        n_repeat=20
    ),
    inp_size=64,
    scale_max=4,
    augment=True,
    sample_q=1024,
)

test_dataset = dict(
    type='SRImplicitDownsampled',
    dataset=dict(
        type='DIV2K',
        root='./sr_dataset/DIV2K_valid_HR'
    ),
    inp_size=64,
    scale_min=2,
    scale_max=4,
    # sample_q=2048,
)

batch_size=16
epoch=100

optimizer = dict(
    type='Adam',
    lr=1e-4
)

encoder=dict(
    type='StyleGAN2',
    arch=dict(type="EncoderDefault", downsample="bilinear"),
    size=64,
    style_dim=512,
    rgb_dim=32
)

model = dict(
    type='LIIF',
    imnet_in_dim=32,
    use_pos_encoding=True,
)

discriminator = dict(
    type='UNet',
    num_in_ch=3
)

discriminator_gradient_norm = True
