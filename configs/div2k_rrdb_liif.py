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
    scale_min=4,
    scale_max=6,
    # sample_q=2048,
)

batch_size=16
epoch=100

optimizer = dict(
    type='Adam',
    lr=1e-4
)

encoder=dict(
    type="RRDBNet",
    num_in_ch=3,
    num_out_ch=32,
    num_feat=64,
    num_block=23,
    scale=1
)

model = dict(
    type='LIIF',
    imnet_in_dim=32
)

discriminator = dict(
    type='UNetSN',
    num_in_ch=3
)

discriminator_gradient_norm = False
